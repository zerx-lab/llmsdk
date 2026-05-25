//! `Anthropic` Skills API.
//!
//! Mirrors `@ai-sdk/anthropic/src/skills/anthropic-skills.ts`. Wraps
//! `POST /v1/skills` (multipart upload) and the followup
//! `GET /v1/skills/{id}/versions/{v}` used to resolve `name` / `description`
//! when the upload response only reports `latest_version`.
//!
//! Beta header: `anthropic-beta: skills-2025-10-02`.
// Rust guideline compliant 2026-02-21

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use llmsdk_provider::error::Result;
use llmsdk_provider::{SkillsModel, UploadSkillOptions, UploadSkillResult};
use llmsdk_provider_utils::http::{RawRequest, get_json, post_raw};
use llmsdk_provider_utils::multipart::Multipart;
use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::auth::apply_request_auth;
use crate::config::Inner;
use crate::error::rewrite_anthropic_error;
use crate::files::upload_data_to_bytes;
use crate::skills::wire::{WireSkillResponse, WireSkillVersionResponse};

const SKILLS_BETA_HEADER: &str = "skills-2025-10-02";

/// `Anthropic` `Skills` API handle.
///
/// Returned by [`crate::Anthropic::skills`]. Implements
/// [`SkillsModel`] for `POST /v1/skills`.
#[derive(Debug, Clone)]
pub struct AnthropicSkills {
    inner: Arc<Inner>,
    provider: String,
}

impl AnthropicSkills {
    pub(crate) fn new(inner: Arc<Inner>, provider: String) -> Self {
        Self { inner, provider }
    }

    fn endpoint(&self) -> String {
        format!("{}/skills", self.inner.base_url)
    }

    fn version_endpoint(&self, skill_id: &str, version: &str) -> String {
        format!(
            "{}/skills/{skill_id}/versions/{version}",
            self.inner.base_url
        )
    }

    fn headers_with_beta(&self) -> HashMap<String, Option<String>> {
        let mut h = self.inner.headers.clone();
        h.insert("anthropic-beta".into(), Some(SKILLS_BETA_HEADER.to_owned()));
        h
    }
}

#[async_trait]
impl SkillsModel for AnthropicSkills {
    fn provider(&self) -> &str {
        &self.provider
    }

    async fn upload_skill(&self, options: UploadSkillOptions) -> Result<UploadSkillResult> {
        let mut mp = Multipart::new();
        if let Some(title) = &options.display_title {
            mp.text("display_title", title);
        }
        for file in &options.files {
            let bytes = upload_data_to_bytes(&file.data)?;
            mp.file("files[]", &file.path, None, &bytes);
        }
        let (boundary, body) = mp.finish();
        let content_type = format!("multipart/form-data; boundary={boundary}");

        let headers = self.headers_with_beta();
        let mut req = RawRequest::new(self.endpoint(), body, content_type.clone());
        req.headers = headers.clone();
        apply_request_auth(
            self.inner.request_auth.as_ref(),
            &mut req.headers,
            "POST",
            &req.url,
            &req.body,
            Some(content_type.as_str()),
        )
        .await?;

        let envelope = match post_raw::<WireSkillResponse>(&self.inner.http, req).await {
            Ok(r) => r,
            Err(err) => return Err(rewrite_anthropic_error(err)),
        };
        let resp = envelope.value;

        // Resolve final name / description: upstream first tries the
        // version-metadata endpoint, then falls back to the upload response.
        let version_meta = if let Some(version) = &resp.latest_version {
            let url = self.version_endpoint(&resp.id, version);
            let mut get_headers = headers.clone();
            apply_request_auth(
                self.inner.request_auth.as_ref(),
                &mut get_headers,
                "GET",
                &url,
                &[],
                None,
            )
            .await?;
            match get_json::<WireSkillVersionResponse, _>(&self.inner.http, &url, &get_headers)
                .await
            {
                Ok(env) => Some(env.value),
                Err(err) => return Err(rewrite_anthropic_error(err)),
            }
        } else {
            None
        };

        let mut provider_reference = HashMap::new();
        provider_reference.insert("anthropic".to_owned(), resp.id.clone());

        let mut meta_obj = JsonMap::new();
        meta_obj.insert("source".to_owned(), JsonValue::String(resp.source.clone()));
        meta_obj.insert(
            "createdAt".to_owned(),
            JsonValue::String(resp.created_at.clone()),
        );
        meta_obj.insert(
            "updatedAt".to_owned(),
            JsonValue::String(resp.updated_at.clone()),
        );
        let mut provider_metadata = HashMap::new();
        provider_metadata.insert("anthropic".to_owned(), meta_obj);

        let name = version_meta
            .as_ref()
            .and_then(|v| v.name.clone())
            .or_else(|| resp.name.clone());
        let description = version_meta
            .as_ref()
            .and_then(|v| v.description.clone())
            .or_else(|| resp.description.clone());

        Ok(UploadSkillResult {
            provider_reference,
            display_title: resp.display_title,
            name,
            description,
            latest_version: resp.latest_version,
            provider_metadata: Some(provider_metadata),
            warnings: Vec::new(),
        })
    }
}
