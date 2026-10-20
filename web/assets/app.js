const $ = (selector) => document.querySelector(selector);
const elements = {
  status: $("#system-status"),
  statusDot: $("#status-dot"),
  setupNotice: $("#setup-notice"),
  setupCopy: $("#setup-copy"),
  fileInput: $("#file-input"),
  dropzone: $("#dropzone"),
  uploadFeedback: $("#upload-feedback"),
  documentList: $("#document-list"),
  docCount: $("#doc-count"),
  refresh: $("#refresh-documents"),
  queryForm: $("#query-form"),
  question: $("#question"),
  askButton: $("#ask-button"),
  queryError: $("#query-error"),
  results: $("#results"),
  answer: $("#answer-text"),
  verificationBadge: $("#verification-badge"),
  verificationLabel: $("#verification-label"),
  answerMeta: $("#answer-meta"),
  answerLatency: $("#answer-latency"),
  evidenceCount: $("#evidence-count"),
  citationList: $("#citation-list"),
  retrievalDetails: $("#retrieval-details"),
  retrievalCount: $("#retrieval-count"),
  retrievedList: $("#retrieved-list"),
};

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

function formatBytes(bytes) {
  if (!bytes) return "0 bytes";
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function showInlineFeedback(message, kind = "") {
  elements.uploadFeedback.textContent = message;
  elements.uploadFeedback.className = `inline-feedback visible ${kind}`;
}

function setSystemStatus(status) {
  const online = status.status === "ok";
  const missing = new Set();
  if (!status.embedding_configured) missing.add("OPENAI_API_KEY");
  if (!status.llm_configured) missing.add(status.llm_provider === "anthropic" ? "ANTHROPIC_API_KEY" : "OPENAI_API_KEY");
  elements.statusDot.className = `status-dot ${online && missing.size === 0 ? "online" : "warning"}`;
  elements.status.textContent = online
    ? (missing.size ? "Setup required" : `${status.llm_model} · ready`)
    : "Storage needs attention";
  elements.setupNotice.hidden = missing.size === 0;
  if (missing.size) {
    elements.setupCopy.textContent = `Add ${[...missing].join(" and ")} to your .env file, then restart the app.`;
  }
}

async function readError(response) {
  try {
    const body = await response.json();
    return typeof body.detail === "string" ? body.detail : "Something went wrong. Please try again.";
  } catch {
    return "Something went wrong. Please try again.";
  }
}

async function loadStatus() {
  try {
    const response = await fetch("/api/v1/status");
    if (!response.ok) throw new Error("status unavailable");
    setSystemStatus(await response.json());
  } catch {
    elements.status.textContent = "Workspace unavailable";
    elements.statusDot.className = "status-dot warning";
    elements.setupNotice.hidden = false;
    elements.setupCopy.textContent = "The API could not be reached. Check that the service is running and refresh this page.";
  }
}

function renderDocuments(documents) {
  elements.docCount.textContent = String(documents.length);
  if (!documents.length) {
    elements.documentList.innerHTML = '<div class="empty-library"><span class="empty-dash"></span><p>Your library is ready.<br />Add a document to get started.</p></div>';
    return;
  }
  elements.documentList.innerHTML = documents.map((document) => {
    const extension = (document.filename.split(".").pop() || "DOC").slice(0, 4).toUpperCase();
    const pages = document.page_count === 1 ? "1 page" : `${document.page_count} pages`;
    const chunks = document.chunk_count === 1 ? "1 chunk" : `${document.chunk_count} chunks`;
    return `<article class="document-item">
      <div class="doc-type" aria-hidden="true">${escapeHtml(extension)}</div>
      <div class="doc-details"><div class="doc-name" title="${escapeHtml(document.filename)}">${escapeHtml(document.filename)}</div><div class="doc-meta">${escapeHtml(pages)} · ${escapeHtml(chunks)}</div></div>
      <button class="delete-doc" type="button" data-delete="${escapeHtml(document.id)}" aria-label="Remove ${escapeHtml(document.filename)}" title="Remove document">×</button>
    </article>`;
  }).join("");
}

async function loadDocuments() {
  try {
    const response = await fetch("/api/v1/documents");
    if (!response.ok) throw new Error(await readError(response));
    renderDocuments(await response.json());
  } catch (error) {
    elements.documentList.innerHTML = `<div class="empty-library"><span class="empty-dash"></span><p>${escapeHtml(error.message || "Could not load the document library.")}</p></div>`;
  }
}

async function uploadFiles(files) {
  const pending = [...files];
  if (!pending.length) return;
  elements.fileInput.value = "";
  let completed = 0;
  for (const file of pending) {
    showInlineFeedback(`Indexing ${file.name}… (${formatBytes(file.size)})`);
    const form = new FormData();
    form.append("file", file);
    try {
      const response = await fetch("/api/v1/documents", { method: "POST", body: form });
      if (!response.ok) throw new Error(await readError(response));
      completed += 1;
      showInlineFeedback(`${file.name} is indexed · ${completed} of ${pending.length} complete`);
    } catch (error) {
      showInlineFeedback(`${file.name}: ${error.message || "upload failed"}`, "error");
    }
  }
  await Promise.all([loadDocuments(), loadStatus()]);
  if (completed && completed === pending.length) {
    showInlineFeedback(`${completed === 1 ? "Document is" : `${completed} documents are`} ready to search.`);
  }
}

function citationMarkup(answer, citations) {
  const knownIds = new Set(citations.map((citation) => citation.source_id));
  const safe = escapeHtml(answer);
  return safe.replace(/\[(S[1-9]\d*)\]/g, (marker, sourceId) => {
    if (!knownIds.has(sourceId)) return marker;
    return `<button class="inline-citation" type="button" data-citation="${sourceId}" aria-label="View source ${sourceId}">${sourceId}</button>`;
  }).replace(/\n/g, "<br />");
}

function renderAnswer(data) {
  elements.results.hidden = false;
  elements.queryError.hidden = true;
  elements.answer.innerHTML = citationMarkup(data.answer, data.citations || []);
  const verified = Boolean(data.citation_verified);
  elements.verificationBadge.classList.toggle("unverified", !verified);
  elements.verificationLabel.textContent = data.abstained ? "SAFE ABSTENTION" : verified ? "CITATIONS VERIFIED" : "CHECK CITATIONS";
  elements.answerMeta.textContent = data.abstained
    ? "No supported answer was found in the indexed documents."
    : data.verification_method === "extractive_fallback"
      ? "Extracted directly from a matching source passage."
      : "Grounded in your indexed documents.";
  elements.answerLatency.textContent = `${data.latency_ms} ms`;

  const citations = data.citations || [];
  elements.evidenceCount.textContent = `${citations.length} ${citations.length === 1 ? "SOURCE" : "SOURCES"}`;
  if (!citations.length) {
    elements.citationList.innerHTML = '<div class="no-citations">No source could be verified for this question. Try adding a relevant document or asking in a different way.</div>';
  } else {
    elements.citationList.innerHTML = citations.map((citation) => {
      const page = citation.page_number ? `PAGE ${citation.page_number}` : `CHUNK ${citation.chunk_index + 1}`;
      return `<article class="citation-card" id="evidence-${escapeHtml(citation.source_id)}">
        <div class="citation-card-top"><div class="citation-source"><span class="source-token">${escapeHtml(citation.source_id)}</span><span class="citation-filename" title="${escapeHtml(citation.filename)}">${escapeHtml(citation.filename)}</span></div><span class="page-label">${escapeHtml(page)}</span></div>
        <p class="citation-quote">${escapeHtml(citation.quote)}</p>
      </article>`;
    }).join("");
  }

  const citationsById = new Set(citations.map((citation) => citation.source_id));
  const otherSources = (data.sources || []).filter((source) => !citationsById.has(source.source_id));
  elements.retrievalDetails.hidden = otherSources.length === 0;
  elements.retrievalCount.textContent = otherSources.length ? `(${otherSources.length})` : "";
  elements.retrievedList.innerHTML = otherSources.map((source) => {
    const page = source.page_number ? `Page ${source.page_number}` : `Chunk ${source.chunk_index + 1}`;
    return `<article class="retrieved-item"><div class="retrieved-item-header"><span>${escapeHtml(source.source_id)} · ${escapeHtml(source.filename)}</span><span>${escapeHtml(page)}</span></div><p>${escapeHtml(source.snippet)}</p></article>`;
  }).join("");
  elements.results.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function ask(question) {
  const normalized = question.trim();
  if (normalized.length < 3) {
    elements.queryError.textContent = "Please enter a little more detail so Atlas can search your documents.";
    elements.queryError.hidden = false;
    return;
  }
  elements.queryError.hidden = true;
  elements.askButton.disabled = true;
  elements.askButton.classList.add("loading");
  elements.askButton.querySelector(".button-label").textContent = "Searching";
  try {
    const response = await fetch("/api/v1/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: normalized }),
    });
    if (!response.ok) throw new Error(await readError(response));
    renderAnswer(await response.json());
  } catch (error) {
    elements.queryError.textContent = error.message || "The search could not be completed. Please try again.";
    elements.queryError.hidden = false;
  } finally {
    elements.askButton.disabled = false;
    elements.askButton.classList.remove("loading");
    elements.askButton.querySelector(".button-label").textContent = "Ask Atlas";
  }
}

elements.fileInput.addEventListener("change", (event) => uploadFiles(event.target.files));
elements.refresh.addEventListener("click", () => loadDocuments());
elements.queryForm.addEventListener("submit", (event) => {
  event.preventDefault();
  ask(elements.question.value);
});
elements.question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    ask(elements.question.value);
  }
});
document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => {
    elements.question.value = button.dataset.question || "";
    elements.question.focus();
  });
});
elements.documentList.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-delete]");
  if (!button) return;
  if (!window.confirm("Remove this document and its index?")) return;
  button.disabled = true;
  try {
    const response = await fetch(`/api/v1/documents/${encodeURIComponent(button.dataset.delete)}`, { method: "DELETE" });
    if (!response.ok) throw new Error(await readError(response));
    await Promise.all([loadDocuments(), loadStatus()]);
  } catch (error) {
    showInlineFeedback(error.message || "Could not remove this document.", "error");
  }
});
elements.citationList.addEventListener("click", (event) => {
  const button = event.target.closest("[data-citation]");
  if (!button) return;
  document.getElementById(`evidence-${button.dataset.citation}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
});

for (const eventName of ["dragenter", "dragover"]) {
  elements.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropzone.classList.add("drag-active");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  elements.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropzone.classList.remove("drag-active");
  });
}
elements.dropzone.addEventListener("drop", (event) => uploadFiles(event.dataTransfer.files));

Promise.all([loadStatus(), loadDocuments()]);
