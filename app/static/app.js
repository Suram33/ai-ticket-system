// Single-page UI for the AI Ticket Desk. Plain JavaScript, no build step.
const state = { tickets: [], fields: [], selectedId: null };

const $ = (sel, root = document) => root.querySelector(sel);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtDate = (iso) => (iso ? new Date(iso).toLocaleString() : "");

async function api(method, url, body) {
  const res = await fetch(url, { method, headers: body ? { "Content-Type": "application/json" } : {}, body: body ? JSON.stringify(body) : undefined });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(formatError(data) || `Request failed (${res.status})`);
    err.data = data;
    throw err;
  }
  return data;
}

function formatError(data) {
  if (data.errors) return Object.values(data.errors).join("\n");
  if (Array.isArray(data.detail)) return data.detail.map((d) => `${d.loc.slice(-1)[0]}: ${d.msg}`).join("\n");
  return data.detail;
}

function toast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(toast.t);
  toast.t = setTimeout(() => el.classList.add("hidden"), 3500);
}

// ---------------- system status ----------------
async function loadStatus() {
  try {
    const h = await api("GET", "/api/health");
    const ml = h.ml.available ? `<span class="pill ok">ML model ready (${h.ml.metadata.n_samples} samples)</span>` : `<span class="pill bad" title="${esc(h.ml.error)}">ML model unavailable</span>`;
    let llm;
    if (!h.llm.enabled) llm = `<span class="pill warn">LLM disabled, template mode</span>`;
    else if (h.llm.reachable && h.llm.model_available) llm = `<span class="pill ok">LLM ${esc(h.llm.model)}</span>`;
    else llm = `<span class="pill warn" title="${esc(h.llm.detail)}">LLM offline, template fallback</span>`;
    const rag = `<span class="pill ok" title="${esc(h.rag.documents.join(", "))}">KB: ${h.rag.chunks} chunks, ${esc(h.rag.embedding_backend)}</span>`;
    $("#system-status").innerHTML = ml + llm + rag;
  } catch (e) {
    $("#system-status").innerHTML = `<span class="pill bad">API unreachable</span>`;
  }
}

// ---------------- ticket list ----------------
async function loadTickets() {
  const p = new URLSearchParams();
  if ($("#filter-q").value) p.set("q", $("#filter-q").value);
  if ($("#filter-status").value) p.set("status", $("#filter-status").value);
  if ($("#filter-priority").value) p.set("priority", $("#filter-priority").value);
  if ($("#filter-inactive").checked) p.set("include_inactive", "true");
  state.tickets = await api("GET", `/api/tickets?${p}`);
  const list = $("#ticket-list");
  if (!state.tickets.length) {
    list.innerHTML = `<li class="muted">No tickets yet.</li>`;
    return;
  }
  list.innerHTML = state.tickets.map((t) => `
    <li data-id="${t.id}" class="${t.id === state.selectedId ? "selected" : ""} ${t.is_active ? "" : "inactive"}">
      <div class="t-subject">#${t.id} ${esc(t.subject)}</div>
      <div class="t-meta"><span>${esc(t.customer_name)}</span><span>${esc(t.status)}</span><span>${esc(t.priority)}</span>
        ${t.category ? `<span>${esc(t.category)}</span>` : ""}${t.is_active ? "" : "<span>deleted</span>"}</div>
    </li>`).join("");
}

async function loadFields() {
  state.fields = await api("GET", "/api/fields");
}

// ---------------- ticket form (create / edit) ----------------
function dynamicFieldInput(f, value) {
  const id = `cf-${f.name}`;
  const req = f.required ? "required" : "";
  const label = `${esc(f.label)}${f.required ? " *" : ""}`;
  const help = f.help_text ? `<span class="muted">${esc(f.help_text)}</span>` : "";
  const err = `<span class="field-error" data-error-for="${f.name}"></span>`;
  switch (f.field_type) {
    case "dropdown":
      return `<label>${label}<select id="${id}" name="${f.name}" data-cf="dropdown" ${req}><option value="">Select...</option>${(f.options || []).map((o) => `<option ${o === value ? "selected" : ""}>${esc(o)}</option>`).join("")}</select>${help}${err}</label>`;
    case "number":
      return `<label>${label}<input id="${id}" name="${f.name}" data-cf="number" type="number" step="any" value="${esc(value)}" ${req}>${help}${err}</label>`;
    case "date":
      return `<label>${label}<input id="${id}" name="${f.name}" data-cf="date" type="date" value="${esc(value)}" ${req}>${help}${err}</label>`;
    case "checkbox":
      return `<label class="inline"><input id="${id}" name="${f.name}" data-cf="checkbox" type="checkbox" ${value ? "checked" : ""}> ${label}${help}${err}</label>`;
    default:
      return `<label>${label}<input id="${id}" name="${f.name}" data-cf="text" value="${esc(value)}" ${req}>${help}${err}</label>`;
  }
}

async function showTicketForm(ticket = null) {
  await loadFields();
  state.selectedId = ticket ? ticket.id : null;
  const t = ticket || { customer_name: "", subject: "", description: "", category: "", priority: "Medium", status: "Open", custom_fields: {} };
  const opt = (list, v) => list.map((o) => `<option ${o === v ? "selected" : ""}>${o}</option>`).join("");
  $("#detail-pane").innerHTML = `
    <h2>${ticket ? `Edit ticket #${ticket.id}` : "New ticket"}</h2>
    <form id="ticket-form" class="grid-form" novalidate>
      <label>Customer name *<input name="customer_name" required value="${esc(t.customer_name)}"><span class="field-error" data-error-for="customer_name"></span></label>
      <label>Subject *<input name="subject" required value="${esc(t.subject)}"><span class="field-error" data-error-for="subject"></span></label>
      <label class="full">Description<textarea name="description" style="min-height:100px">${esc(t.description)}</textarea></label>
      <label>Category<input name="category" list="categories" value="${esc(t.category)}" placeholder="Leave empty to let the ML model predict it">
        <datalist id="categories"><option>Account</option><option>Billing</option><option>Shipping</option><option>Returns</option><option>Technical</option></datalist></label>
      <label>Priority<select name="priority">${opt(["Low", "Medium", "High", "Urgent"], t.priority)}</select></label>
      ${ticket ? `<label>Status<select name="status">${opt(["Open", "In Progress", "Resolved", "Closed"], t.status)}</select></label>` : ""}
      ${state.fields.length ? `<div class="full"><h3>Additional details</h3></div>` : ""}
      ${state.fields.map((f) => dynamicFieldInput(f, (t.custom_fields || {})[f.name])).join("")}
      <div class="form-actions">
        <button class="primary" type="submit">${ticket ? "Save changes" : "Create ticket and generate solution"}</button>
        <button type="button" id="btn-cancel">Cancel</button>
      </div>
      <div class="form-error" id="ticket-form-error"></div>
    </form>`;
  $("#btn-cancel").onclick = () => (ticket ? selectTicket(ticket.id) : ($("#detail-pane").innerHTML = `<div class="empty">Select a ticket or create a new one.</div>`));
  $("#ticket-form").onsubmit = (e) => submitTicketForm(e, ticket);
}

function readTicketForm(form) {
  const fd = new FormData(form);
  const body = { customer_name: fd.get("customer_name"), subject: fd.get("subject"), description: fd.get("description"),
    category: fd.get("category") || null, priority: fd.get("priority"), custom_fields: {} };
  if (fd.has("status")) body.status = fd.get("status");
  form.querySelectorAll("[data-cf]").forEach((el) => {
    body.custom_fields[el.name] = el.dataset.cf === "checkbox" ? el.checked : el.value;
  });
  return body;
}

function validateClient(form, body) {
  // Quick checks in the browser; the server validates again.
  const errors = {};
  if (!body.customer_name.trim()) errors.customer_name = "Customer name is required";
  if (!body.subject.trim()) errors.subject = "Subject is required";
  state.fields.forEach((f) => {
    if (f.required && f.field_type !== "checkbox" && !String(body.custom_fields[f.name] ?? "").trim()) errors[f.name] = `${f.label} is required`;
  });
  showFieldErrors(form, errors);
  return !Object.keys(errors).length;
}

function showFieldErrors(form, errors) {
  form.querySelectorAll("[data-error-for]").forEach((el) => (el.textContent = errors[el.dataset.errorFor] || ""));
}

async function submitTicketForm(e, ticket) {
  e.preventDefault();
  const form = e.target;
  const body = readTicketForm(form);
  $("#ticket-form-error").textContent = "";
  if (!validateClient(form, body)) return;
  const btn = form.querySelector("button[type=submit]");
  btn.disabled = true;
  try {
    if (ticket) {
      const updated = await api("PUT", `/api/tickets/${ticket.id}`, body);
      toast("Ticket updated");
      await loadTickets();
      renderDetail(updated);
    } else {
      // Save + ML prediction first, then run the slower RAG/LLM step with a visible spinner.
      const created = await api("POST", "/api/tickets?generate=false", body);
      state.selectedId = created.id;
      await loadTickets();
      renderDetail(created, { generating: true });
      await generate(created.id);
    }
  } catch (err) {
    if (err.data && err.data.errors) showFieldErrors(form, err.data.errors);
    $("#ticket-form-error").textContent = err.message;
    btn.disabled = false;
  }
}

// ---------------- ticket detail ----------------
async function selectTicket(id) {
  state.selectedId = id;
  document.querySelectorAll("#ticket-list li").forEach((li) => li.classList.toggle("selected", Number(li.dataset.id) === id));
  renderDetail(await api("GET", `/api/tickets/${id}`));
}

function renderDetail(t, opts = {}) {
  const custom = Object.entries(t.custom_fields || {});
  const fieldLabel = (name) => (state.fields.find((f) => f.name === name) || {}).label || name;
  const fmtVal = (v) => (v === true ? "Yes" : v === false ? "No" : esc(v));
  const grounding = t.grounding_report;
  const sources = t.rag_sources || [];
  $("#detail-pane").innerHTML = `
    <div class="detail-head">
      <div>
        <h2>#${t.id} ${esc(t.subject)}</h2>
        <div class="badges"><span class="badge ${esc(t.status)}">${esc(t.status)}</span><span class="badge ${esc(t.priority)}">${esc(t.priority)}</span>
          ${t.category ? `<span class="badge">${esc(t.category)}</span>` : ""}${t.is_active ? "" : `<span class="badge High">deleted</span>`}</div>
      </div>
      <div class="actions">
        <select id="quick-status" title="Change status">${["Open", "In Progress", "Resolved", "Closed"].map((s) => `<option ${s === t.status ? "selected" : ""}>${s}</option>`).join("")}</select>
        <button id="btn-edit">Edit</button>
        ${t.is_active ? `<button id="btn-delete" class="danger">Delete</button>` : `<button id="btn-restore">Restore</button><button id="btn-hard-delete" class="danger">Delete permanently</button>`}
      </div>
    </div>

    <div class="section">
      <dl class="kv">
        <dt>Customer</dt><dd>${esc(t.customer_name)}</dd>
        <dt>Description</dt><dd>${esc(t.description) || '<span class="muted">(none)</span>'}</dd>
        ${custom.map(([k, v]) => `<dt>${esc(fieldLabel(k))}</dt><dd>${fmtVal(v)}</dd>`).join("")}
        <dt>Created</dt><dd>${fmtDate(t.created_at)}</dd><dt>Updated</dt><dd>${fmtDate(t.updated_at)}</dd>
      </dl>
    </div>

    <div class="section">
      <h3><span class="step">ML</span> Predicted one-line solution <button id="btn-repredict" style="margin-left:auto">Re-run prediction</button></h3>
      ${t.prediction_error ? `<div class="notice bad">${esc(t.prediction_error)}</div>` : ""}
      ${t.predicted_solution ? `<div class="prediction"><div class="solution">${esc(t.predicted_solution)}</div>
        <div class="muted">Confidence ${(t.prediction_confidence * 100).toFixed(0)}% · predicted category ${esc(t.predicted_category)}</div></div>` : (t.prediction_error ? "" : `<div class="muted">No prediction yet.</div>`)}
    </div>

    <div class="section">
      <h3><span class="step">RAG</span> Knowledge-base content used</h3>
      ${sources.length ? `<div class="sources">${sources.map((s) => `<details class="source"><summary>${esc(s.title)} › ${esc(s.section)} <span class="score">(similarity ${s.score.toFixed(2)}, ${esc(s.source)})</span></summary><p>${esc(s.text)}</p></details>`).join("")}</div>`
        : `<div class="muted">${t.generated_at ? "No relevant knowledge-base content found." : "Not generated yet."}</div>`}
    </div>

    <div class="section">
      <h3><span class="step">LLM</span> Detailed response
        ${t.response_status !== "none" ? `<span class="badge ${esc(t.response_status)}">${esc(t.response_status)}</span>` : ""}
        <button id="btn-generate" style="margin-left:auto">${t.generated_at ? "Regenerate" : "Generate"}</button></h3>
      <div id="gen-status">${opts.generating ? `<span class="spinner"></span> Retrieving knowledge and generating a response with the local LLM...` : ""}</div>
      ${t.generation_error ? `<div class="notice warn">${esc(t.generation_error)}</div>` : ""}
      ${t.response_source ? `<div class="muted">Generated by ${t.response_source === "llm" ? `local LLM (${esc(t.response_model)})` : "knowledge-base template (LLM fallback)"} · ${fmtDate(t.generated_at)}</div>` : ""}
      ${grounding && grounding.unsupported && grounding.unsupported.length ? `<div class="notice warn"><b>Review:</b> these sentences are not clearly backed by the knowledge base:<ul>${grounding.unsupported.map((u) => `<li>${esc(u.sentence)}</li>`).join("")}</ul></div>` : ""}
      <textarea id="final-response" placeholder="The generated response appears here. You can edit it before saving.">${esc(t.final_response || "")}</textarea>
      <div class="actions" style="margin-top:8px">
        <button id="btn-save-response">Save draft</button>
        <button id="btn-approve" class="primary">Approve and save</button>
        ${t.ai_response && t.final_response !== t.ai_response ? `<button id="btn-reset-response">Restore AI version</button>` : ""}
      </div>
    </div>`;

  $("#btn-edit").onclick = () => showTicketForm(t);
  $("#quick-status").onchange = async (e) => { renderDetail(await api("PUT", `/api/tickets/${t.id}`, { status: e.target.value })); loadTickets(); toast("Status updated"); };
  if ($("#btn-delete")) $("#btn-delete").onclick = async () => { if (!confirm("Delete this ticket? It can be restored later.")) return; renderDetail(await api("DELETE", `/api/tickets/${t.id}`)); loadTickets(); toast("Ticket deleted (deactivated)"); };
  if ($("#btn-restore")) $("#btn-restore").onclick = async () => { renderDetail(await api("POST", `/api/tickets/${t.id}/restore`)); loadTickets(); toast("Ticket restored"); };
  if ($("#btn-hard-delete")) $("#btn-hard-delete").onclick = async () => {
    if (!confirm("Permanently delete this ticket? This cannot be undone.")) return;
    await api("DELETE", `/api/tickets/${t.id}?hard=true`);
    state.selectedId = null; $("#detail-pane").innerHTML = `<div class="empty">Ticket deleted.</div>`; loadTickets();
  };
  $("#btn-repredict").onclick = async () => { renderDetail(await api("POST", `/api/tickets/${t.id}/predict`)); toast("Prediction updated"); };
  $("#btn-generate").onclick = () => {
    if (t.response_status === "approved" && !confirm("The approved response will be kept; the new AI draft is shown under 'Restore AI version'. Continue?")) return;
    generate(t.id);
  };
  const saveResp = async (approve) => {
    try {
      renderDetail(await api("PUT", `/api/tickets/${t.id}/response`, { final_response: $("#final-response").value, approve }));
      toast(approve ? "Response approved and saved" : "Draft saved");
    } catch (err) { toast(err.message); }
  };
  $("#btn-save-response").onclick = () => saveResp(false);
  $("#btn-approve").onclick = () => saveResp(true);
  if ($("#btn-reset-response")) $("#btn-reset-response").onclick = () => ($("#final-response").value = t.ai_response);
}

async function generate(id) {
  const btn = $("#btn-generate");
  if (btn) btn.disabled = true;
  $("#gen-status").innerHTML = `<span class="spinner"></span> Retrieving knowledge and generating a response with the local LLM...`;
  try {
    const t = await api("POST", `/api/tickets/${id}/generate`);
    if (state.selectedId === id) renderDetail(t);
  } catch (err) {
    $("#gen-status").innerHTML = `<div class="notice bad">${esc(err.message)}</div>`;
    if (btn) btn.disabled = false;
  }
}

// ---------------- custom fields admin ----------------
async function renderFieldsAdmin() {
  const fields = await api("GET", "/api/fields?include_inactive=true");
  $("#fields-body").innerHTML = fields.map((f) => `
    <tr>
      <td>${esc(f.label)}</td><td><code>${esc(f.name)}</code></td><td>${esc(f.field_type)}</td>
      <td><input type="checkbox" data-req="${f.id}" ${f.required ? "checked" : ""}></td>
      <td>${esc((f.options || []).join(", "))}</td>
      <td>${f.is_active ? "Yes" : "No"}</td>
      <td><button data-toggle="${f.id}" data-active="${f.is_active}">${f.is_active ? "Deactivate" : "Activate"}</button></td>
    </tr>`).join("");
  document.querySelectorAll("[data-req]").forEach((el) => (el.onchange = async () => { await api("PUT", `/api/fields/${el.dataset.req}`, { required: el.checked }); toast("Field updated"); }));
  document.querySelectorAll("[data-toggle]").forEach((el) => (el.onclick = async () => {
    await api("PUT", `/api/fields/${el.dataset.toggle}`, { is_active: el.dataset.active !== "true" });
    renderFieldsAdmin();
  }));
}

$("#field-form").onsubmit = async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const body = { name: fd.get("name").trim(), label: fd.get("label").trim(), field_type: fd.get("field_type"), required: fd.get("required") === "on",
    help_text: fd.get("help_text") || null, options: fd.get("options") ? fd.get("options").split(",").map((s) => s.trim()).filter(Boolean) : null };
  try {
    await api("POST", "/api/fields", body);
    e.target.reset();
    $("#field-form-error").textContent = "";
    toast("Field added. It now appears on the ticket form.");
    renderFieldsAdmin();
  } catch (err) { $("#field-form-error").textContent = err.message; }
};

// ---------------- wiring ----------------
document.querySelectorAll(".tab").forEach((tab) => (tab.onclick = () => {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
  $("#view-tickets").classList.toggle("hidden", tab.dataset.view !== "tickets");
  $("#view-fields").classList.toggle("hidden", tab.dataset.view !== "fields");
  if (tab.dataset.view === "fields") renderFieldsAdmin();
}));
$("#btn-new").onclick = () => showTicketForm();
$("#ticket-list").onclick = (e) => { const li = e.target.closest("li[data-id]"); if (li) selectTicket(Number(li.dataset.id)); };
let searchTimer;
$("#filter-q").oninput = () => { clearTimeout(searchTimer); searchTimer = setTimeout(loadTickets, 250); };
["#filter-status", "#filter-priority", "#filter-inactive"].forEach((s) => ($(s).onchange = loadTickets));

loadStatus();
loadFields().then(loadTickets);
