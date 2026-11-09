# ModelMint Feature Walkthrough

Total runtime: 4m 35s

| Time | Feature | Narration |
|---|---|---|
| 00:00 | ModelMint - Feature Walkthrough | Welcome to ModelMint, a secure model as a service platform. In this walkthrough, we will upload a trained model, serve single and batch predictions, create scoped API keys, promote new versions with canary traffic, detect input drift, and monitor every request end to end. |
| 00:17 | 01 - Model Registry | This is the ModelMint registry. It is empty right now, ready for the first model upload. From here you can search, filter, inspect, and delete models. |
| 00:28 | 02 - Scoped API Keys (Locked) | The API keys page starts locked. Administrative actions such as creating or revoking keys require an admin key, so access is never open by default. |
| 00:38 | 03 - Admin Unlock | Unlocking with the bootstrap admin key reveals scoped key management, per-key usage, and the security audit log. The key stays in this browser tab only. |
| 00:48 | 04 - Creating a Scoped Key | Now we create a scoped key for a client. The plaintext secret is shown exactly once, and only an HMAC hash is stored on the server. If it is lost, you rotate it. |
| 00:58 | 05 - Key Scopes and Audit | The client key has read, write, and monitoring scopes, plus predict. The audit log immediately records the key creation with the exact scopes granted. |
| 01:08 | 06 - Upload and Sandbox Validation | Next, upload a scikit-learn pickle. ModelMint detects the framework, validates the artifact in a sandbox, stores a SHA-256 checksum, and automatically creates a live inference endpoint. |
| 01:22 | 07 - Detected Schema | The model detail view shows the detected framework, task type, schema confidence, feature count, class count, and sandbox load time. A synthetic input test confirms the model is servable. |
| 01:34 | 08 - Task-Aware Try It | The Try It panel generates a task-aware example request. For a tabular model it sends a features array, so you can test the endpoint without writing any code. |
| 01:44 | 09 - Single Prediction | A single prediction returns in milliseconds from a persistent sandboxed worker. The API process never loads untrusted model code directly. |
| 01:53 | 10 - Batch Prediction | Batch mode accepts an array of instances and returns one prediction per row. This is useful for scoring small datasets in a single request. |
| 02:02 | 11 - Copy-Ready Integrations | Copy-ready curl and Python snippets make integration immediate. They use your API key and the model's generated endpoint. |
| 02:10 | 12 - Monitoring and Metrics | Monitoring aggregates live request counts, success rate, average and p95 latency, a latency chart, and the most recent inference logs. |
| 02:21 | 13 - Upload a New Version | Upload a new version into the same model family. ModelMint validates it in the sandbox and keeps it inactive, so production traffic is untouched until promotion. |
| 02:31 | 14 - Promote a Release | Promoting version two makes it the active release. Version one remains servable, so a rollback is one click away if the new release misbehaves. |
| 02:41 | 15 - A/B Traffic Split | Traffic weights turn a release into a canary. Here version two and version one each receive half of the requests through the stable family endpoint. |
| 02:49 | 16 - Input Drift Detection | Drift detection compares the recent request window with the baseline and flags features whose distribution has shifted, without ever storing raw request bodies. |
| 02:59 | 17 - Supported Formats API | The formats endpoint reports every supported framework, its file extensions, task types, and whether its optional package is installed in this deployment. |
| 03:10 | 18 - Prometheus Metrics | Prometheus metrics expose model counts by status, inference requests, latency totals, and active API keys, ready to scrape into your observability stack. |
| 03:21 | 19 - Authentication Required | Security is enforced at the API layer. Calling predict without an API key is rejected with 401 Unauthorized, before any model code runs. |
| 03:31 | 20 - Security Audit Log | The audit log tracks key creation, model uploads, promotions, rotations, revocations, and deletions. Every security-sensitive action is attributable and timestamped. |
| 03:43 | 21 - Key Rotation | Rotating a key immediately invalidates the old secret and issues a new plaintext key, shown once. Existing integrations must switch to the new value. |
| 03:53 | 22 - Key Revocation | Revoking a key disables it permanently. The key remains in the table with a revoked status, preserving history for audits. |
| 04:01 | 23 - Model Deletion | Deleting a model removes its stored artifact and inference endpoint. The registry returns to an empty state, ready for the next upload. |
| 04:10 | 24 - Final Audit Trail | Every action remains visible in the audit trail. ModelMint is ready to mint, serve, and govern models with scoped access and full observability. |
| 04:19 | Mint. Serve. Govern. | That is ModelMint. Upload a supported model, serve it instantly, promote new versions with canary traffic, watch for input drift, and govern every action with scoped keys, sandboxed execution, and a complete audit trail. Thanks for watching. |
