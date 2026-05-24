//! Integration tests for HTTP helpers against a wiremock server.
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;

use llmsdk_provider_utils::http::{HttpClient, JsonRequest, post_json};
use serde::{Deserialize, Serialize};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    id: String,
    output: String,
}

fn client() -> HttpClient {
    HttpClient::new().expect("client builds")
}

fn req(url: &str) -> JsonRequest<ChatRequest> {
    JsonRequest::new(
        url,
        ChatRequest {
            model: "test".into(),
            messages: vec!["hi".into()],
        },
    )
    .header("authorization", Some("Bearer test-key".into()))
}

#[tokio::test]
async fn happy_path_200() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat"))
        .and(header("authorization", "Bearer test-key"))
        .and(header("content-type", "application/json"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({ "id": "r-1", "output": "ok" })),
        )
        .mount(&server)
        .await;

    let url = format!("{}/v1/chat", server.uri());
    let response = post_json::<_, ChatResponse>(&client(), req(&url))
        .await
        .expect("ok");
    assert_eq!(response.status.as_u16(), 200);
    assert_eq!(response.value.id, "r-1");
    assert_eq!(response.value.output, "ok");
}

#[tokio::test]
async fn http_429_is_retryable_api_call() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat"))
        .respond_with(
            ResponseTemplate::new(429).set_body_string(r#"{"error":{"message":"rate limited"}}"#),
        )
        .mount(&server)
        .await;

    let url = format!("{}/v1/chat", server.uri());
    let err = post_json::<_, ChatResponse>(&client(), req(&url))
        .await
        .expect_err("should error");
    assert!(err.is_api_call());
    assert!(err.is_retryable(), "429 should be retryable");
    assert_eq!(err.status_code(), Some(429));
}

#[tokio::test]
async fn http_400_is_not_retryable() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat"))
        .respond_with(ResponseTemplate::new(400).set_body_string("bad"))
        .mount(&server)
        .await;

    let url = format!("{}/v1/chat", server.uri());
    let err = post_json::<_, ChatResponse>(&client(), req(&url))
        .await
        .expect_err("should error");
    assert!(err.is_api_call());
    assert!(!err.is_retryable());
    assert_eq!(err.status_code(), Some(400));
}

#[tokio::test]
async fn http_503_is_retryable() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&server)
        .await;

    let url = format!("{}/v1/chat", server.uri());
    let err = post_json::<_, ChatResponse>(&client(), req(&url))
        .await
        .expect_err("should error");
    assert!(err.is_retryable());
    assert_eq!(err.status_code(), Some(503));
}

#[tokio::test]
async fn json_parse_failure_on_2xx_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
        .mount(&server)
        .await;

    let url = format!("{}/v1/chat", server.uri());
    let err = post_json::<_, ChatResponse>(&client(), req(&url))
        .await
        .expect_err("should error");
    assert!(!err.is_api_call(), "JSON failure should not be ApiCall");
    assert!(format!("{err}").contains("json"));
}

#[tokio::test]
async fn transport_error_is_retryable() {
    // Use a port that is almost certainly closed.
    let url = "http://127.0.0.1:1/v1/chat".to_owned();
    let err = post_json::<_, ChatResponse>(&client(), req(&url))
        .await
        .expect_err("should error");
    assert!(err.is_api_call());
    assert!(err.is_retryable(), "network failure should be retryable");
}

#[tokio::test]
async fn header_override_works() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat"))
        .and(header("x-llmsdk", "1"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({ "id": "r-2", "output": "ok" })),
        )
        .mount(&server)
        .await;

    let url = format!("{}/v1/chat", server.uri());
    let request = req(&url).header("x-llmsdk", Some("1".into()));
    let response = post_json::<_, ChatResponse>(&client(), request)
        .await
        .expect("ok");
    assert_eq!(response.value.id, "r-2");
}

#[tokio::test]
async fn empty_headers_map_is_fine() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({ "id": "r-3", "output": "ok" })),
        )
        .mount(&server)
        .await;

    let url = format!("{}/v1/chat", server.uri());
    let request = JsonRequest::new(
        &url,
        ChatRequest {
            model: "x".into(),
            messages: vec![],
        },
    );
    let _ = request.headers; // touch field for coverage
    let response = post_json::<_, ChatResponse>(&client(), request)
        .await
        .expect("ok");
    assert_eq!(response.value.id, "r-3");
}

#[tokio::test]
async fn captures_response_headers() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-request-id", "req-abc")
                .set_body_json(serde_json::json!({ "id": "r-4", "output": "ok" })),
        )
        .mount(&server)
        .await;

    let url = format!("{}/v1/chat", server.uri());
    let response: HashMap<String, String> = post_json::<_, ChatResponse>(&client(), req(&url))
        .await
        .expect("ok")
        .headers;
    assert_eq!(
        response.get("x-request-id").map(String::as_str),
        Some("req-abc")
    );
}
