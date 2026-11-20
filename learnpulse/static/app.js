// LearnPulse teacher panel - plain JavaScript, talks to the FastAPI service.
const S = { meta: null, role: "teacher", klass: "8-B", week: 30, tab: "list", list: null, selected: null, detail: null, lang: "en" };
const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const user = () => (S.role === "teacher" ? `teacher:${S.klass}` : S.role);

async function api(path, opts = {}) {
  const res = await fetch(path, { ...opts, headers: { "X-Demo-User": user(), "Content-Type": "application/json", ...(opts.headers || {}) } });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || res.statusText);
  return body;
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(toast._h);
  toast._h = setTimeout(() => t.classList.add("hidden"), 2600);
}

async function init() {
  S.meta = await api("/api/meta");
  S.week = S.meta.demo_week;
  $("#klass").innerHTML = S.meta.classes.map((c) => `<option ${c === S.klass ? "selected" : ""}>${c}</option>`).join("");
  const weeks = [];
  for (let w = S.meta.first_week; w <= S.meta.last_week; w++) weeks.push(`<option value="${w}" ${w === S.week ? "selected" : ""}>Week ${w}</option>`);
  $("#week").innerHTML = weeks.join("");
  $("#role").onchange = (e) => { S.role = e.target.value; refresh(); };
  $("#klass").onchange = (e) => { S.klass = e.target.value; refresh(); };
  $("#week").onchange = (e) => { S.week = +e.target.value; refresh(); };
  document.querySelectorAll("#tabs button").forEach((b) => (b.onclick = () => { S.tab = b.dataset.tab; refresh(); }));
  refresh();
}

function refresh() {
  document.querySelectorAll("#tabs button").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === S.tab);
    if (b.hasAttribute("data-head")) b.hidden = S.role !== "head";
  });
  if (S.tab === "alerts" && S.role !== "head") S.tab = "list";
  $("#klass").disabled = S.role === "counsellor";
  document.querySelectorAll(".view").forEach((v) => v.classList.add("hidden"));
  $(`#view-${S.tab}`).classList.remove("hidden");
  ({ list: loadList, outcomes: loadOutcomes, alerts: loadAlerts, evaluation: loadEvaluation })[S.tab]();
}

// ------------------------------------------------------------------ weekly list
async function loadList() {
  if (S.role === "counsellor") {
    $("#summary").innerHTML = "";
    $("#list").innerHTML = `<div class="panel">Counsellors see only students referred to them. Open <b>Outcomes</b>.</div>`;
    $("#detail").innerHTML = "";
    return;
  }
  try {
    S.list = await api(`/api/list?klass=${encodeURIComponent(S.klass)}&week=${S.week}`);
  } catch (e) {
    $("#list").innerHTML = `<div class="panel">${esc(e.message)}</div>`;
    return;
  }
  const L = S.list;
  if (L.status === "data_incomplete") {
    $("#summary").innerHTML = stat("Attendance marked", Math.round(L.attendance_marked * 100) + "%");
    $("#list").innerHTML = `<div class="incomplete">Data incomplete - ${esc(L.message)}</div>`;
    $("#detail").innerHTML = "";
    return;
  }
  $("#summary").innerHTML =
    stat("Class", L.class) + stat("Week", L.week) + stat("Flagged by rules", L.flagged_in_class) + stat("On this week's list", L.students.length) +
    `<div class="stat note">A student appears only when a rule has fired. The model only sets the order. At most ${S.meta.list_size} per class per week.</div>`;
  $("#list").innerHTML = L.students.map(card).join("") || `<div class="panel">No student needs attention this week.</div>`;
  document.querySelectorAll(".card").forEach((c) => (c.onclick = () => select(c.dataset.sid)));
  const keep = L.students.find((s) => s.student_id === S.selected);
  if (keep) select(keep.student_id, true);
  else if (L.students.length) select(L.students[0].student_id, true);
  else $("#detail").innerHTML = "";
}

const stat = (k, v) => `<div class="stat"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`;
const chip = (s) => `<span class="chip ${s[0]}">${s}</span>`;

function card(s) {
  const top = s.reasons[0];
  const dec = s.decision ? `<span class="state ${s.decision.action}">${{ accept: "Accepted", edit: "Edited and accepted", dismiss: "Dismissed - " + esc(s.decision.dismiss_reason) }[s.decision.action]}</span>` : "";
  const esc8 = s.escalated_from_week ? `<span class="state escalated">No recovery since week ${s.escalated_from_week} - escalate</span>` : "";
  return `<div class="card ${s.student_id === S.selected ? "selected" : ""}" data-sid="${s.student_id}">
    <div class="row1"><div class="rank">${s.rank}</div>
      <div><div class="who">${esc(s.name)}</div><div class="sid">${s.student_id} - ${esc(s.class)}</div></div>
      <div class="prio"><div class="num">${s.priority.toFixed(2)}</div><div class="lbl">priority</div></div></div>
    <div class="bar"><span style="width:${Math.round(s.priority * 100)}%"></span></div>
    <div class="reason">${esc(top ? top.text : "")}</div>
    <div class="chips">${s.signals.map(chip).join("")}${s.annotations.map((a) => chip(a.signal)).join("")}</div>
    ${dec}${esc8}</div>`;
}

async function select(sid, quiet) {
  S.selected = sid;
  document.querySelectorAll(".card").forEach((c) => c.classList.toggle("selected", c.dataset.sid === sid));
  S.detail = await api(`/api/students/${sid}?week=${S.week}`);
  S.lang = S.detail.guardian_language === "hi" ? "hi" : "en";
  renderDetail();
}

function spark(values, opts = {}) {
  const W = 520, H = 90, pad = 6;
  const pts = values.map((v, i) => [i, v]).filter((p) => p[1] !== null);
  if (!pts.length) return "";
  const x = (i) => pad + (i / Math.max(values.length - 1, 1)) * (W - 2 * pad);
  const lo = opts.min ?? 0.3, hi = opts.max ?? 1;
  const y = (v) => H - pad - ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * (H - 2 * pad);
  const path = pts.map((p, k) => `${k ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join("");
  const lines = (opts.lines || []).map((l) => `<line x1="${pad}" x2="${W - pad}" y1="${y(l.v)}" y2="${y(l.v)}" stroke="${l.c}" stroke-dasharray="4 4" stroke-width="1"/>`).join("");
  const w4 = values.length >= 4 ? `<rect x="${x(values.length - 4) - 4}" y="0" width="${W - x(values.length - 4)}" height="${H}" fill="#fdf1df"/>` : "";
  return `<svg class="spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${w4}${lines}<path d="${path}" fill="none" stroke="#10283f" stroke-width="2"/></svg>`;
}

function renderDetail() {
  const d = S.detail;
  const listed = (S.list?.students || []).find((s) => s.student_id === d.student_id) || {};
  const decided = listed.decision;
  const pb = d.playbook;
  const draft = d.parent_drafts[S.lang] || { text: "", source: "template" };
  const scores = Object.entries(d.scores).map(([sub, arr]) =>
    `<tr><td class="s">${esc(sub)}</td>${arr.map((a) => `<td class="${a.score < 40 ? "low" : ""}" title="week ${a.week}">${a.score.toFixed(0)}</td>`).join("")}</tr>`).join("");
  $("#detail").innerHTML = `<div class="panel">
    <div class="dhead"><h2>${esc(d.name)}</h2><span class="meta">${d.student_id} - Class ${esc(d.class)} - week ${d.week} - parent language: ${d.guardian_language === "hi" ? "Hindi" : "English"}</span>
      <span class="mix">priority <b>${d.priority.toFixed(2)}</b> = 0.6 x model <b>${d.model_probability.toFixed(2)}</b> + 0.4 x rules <b>${d.rule_severity.toFixed(2)}</b></span></div>
    <div class="grid2">
      <div><h3>Why this student</h3><ul class="facts">${d.facts.map((f) => `<li><span class="sig ${f.signal[0]}">${f.signal}</span><div>${esc(f.text)}<div class="t">${esc(f.title)}</div></div></li>`).join("")}</ul></div>
      <div><h3>Attendance by week</h3>${spark(d.attendance_trend, { lines: [{ v: 0.9, c: "#1d7a3a" }, { v: 0.75, c: "#b42318" }] })}
        <div class="legend">Shaded: last 4 weeks. Dashed: 90% and 75%.</div>
        <h3 style="margin-top:12px">Assessment scores (%)</h3><table class="scores">${scores}</table></div>
    </div>
    <div class="action">
      <h3>Suggested next step</h3>
      <div class="field"><input id="step" value="${esc(pb.action)}"><span class="hint">Owner: ${esc(pb.owner)} - tracked for 4 weeks: ${esc(pb.tracked)}</span></div>
      <div class="field"><label style="display:flex;align-items:center;gap:10px">Parent note draft
        <span class="langs"><button data-l="en" class="${S.lang === "en" ? "on" : ""}">English</button><button data-l="hi" class="${S.lang === "hi" ? "on" : ""}">Hindi</button></span>
        <span class="verified">Every number checked against the facts</span></label>
        <textarea id="note">${esc(draft.text)}</textarea>
        <span class="hint">Nothing is sent until you approve. Approved notes go through the ERP's own messaging.</span></div>
      <div class="buttons">
        <button class="btn primary" id="b-accept" ${decided ? "disabled" : ""}>Accept</button>
        <button class="btn secondary" id="b-edit" ${decided ? "disabled" : ""}>Save edit and accept</button>
        <span class="dismiss-box"><select id="dreason">${S.meta.dismiss_reasons.map((r) => `<option>${esc(r)}</option>`).join("")}</select>
        <button class="btn ghost" id="b-dismiss" ${decided ? "disabled" : ""}>Dismiss</button></span>
      </div>
    </div></div>`;
  document.querySelectorAll(".langs button").forEach((b) => (b.onclick = () => { S.lang = b.dataset.l; renderDetail(); }));
  $("#b-accept").onclick = () => decide("accept");
  $("#b-edit").onclick = () => decide("edit");
  $("#b-dismiss").onclick = () => decide("dismiss");
}

async function decide(action) {
  const d = S.detail;
  const body = { student_id: d.student_id, week: d.week, action };
  if (action === "edit") { body.step = $("#step").value; body.note = $("#note").value; }
  if (action === "dismiss") body.dismiss_reason = $("#dreason").value;
  try {
    await api("/api/decisions", { method: "POST", body: JSON.stringify(body) });
    toast({ accept: "Accepted - next step assigned, follow-up for 4 weeks", edit: "Edit saved - next step assigned", dismiss: "Dismissed - snoozed for 2 weeks" }[action]);
    await loadList();
  } catch (e) { toast(e.message); }
}

// ------------------------------------------------------------------ outcomes
async function loadOutcomes() {
  const o = await api(`/api/outcomes${S.role === "head" ? "?klass=" + encodeURIComponent(S.klass) : ""}`);
  const rows = o.decisions.map((r) => {
    const cls = { recovered: "recovered", "not recovered": "notrec", pending: "pending", dismissed: "dismissed" }[r.status];
    return `<tr><td><b>${esc(r.name)}</b><div class="legend">${r.student_id} - ${esc(r.class)}</div></td><td>Week ${r.week}</td>
      <td>${esc(r.action)}${r.dismiss_reason ? " - " + esc(r.dismiss_reason) : ""}</td><td>${esc(r.step || "")}</td>
      <td>${r.follow_up_weeks ? "Weeks " + r.follow_up_weeks[0] + "-" + r.follow_up_weeks[1] : ""}</td>
      <td>${esc(r.measure || "")}</td><td><span class="pill ${cls}">${esc(r.status)}</span></td></tr>`;
  }).join("");
  const sum = Object.entries(o.by_playbook_action).map(([k, v]) =>
    `<div class="stat"><div class="k">${esc(k.replace("_", " "))}</div><div class="v">${v.recovered} / ${v.recovered + v["not recovered"]}</div><div class="legend">recovered${v.pending ? ", " + v.pending + " pending" : ""}</div></div>`).join("");
  $("#view-outcomes").innerHTML = `<p class="lead">Every decision is logged. For accepted steps the flagged indicators are followed for four weeks and marked recovered or not recovered - evidence of which action works for which cause. ${esc(o.note)}</p>
    <div class="summary">${stat("Decisions", o.decisions.length)}${stat("Accepted or edited", o.acceptance_rate === null ? "-" : Math.round(o.acceptance_rate * 100) + "%")}${sum}</div>
    <table class="table"><thead><tr><th>Student</th><th>Decided</th><th>Decision</th><th>Step</th><th>Follow-up</th><th>Measured</th><th>Outcome</th></tr></thead>
    <tbody>${rows || `<tr><td colspan="7">No decisions yet.</td></tr>`}</tbody></table>`;
}

// ------------------------------------------------------------------ section alerts
async function loadAlerts() {
  const a = await api(`/api/section-alerts?week=${S.week}`);
  $("#view-alerts").innerHTML = `<p class="lead">A whole-section drop in one subject is routed to the academic head as a possible teaching or assessment issue - forty students are not flagged for one hard test.</p>
    <div class="cards3">${a.map((x) => `<div class="alert ${x.active ? "" : "old"}"><div class="h">${esc(x.class)} - ${esc(x.subject)} <span class="legend">week ${x.week}${x.active ? " - active" : ""}</span></div>
      <div class="t">${esc(x.text)}</div><div class="a">${esc(x.action)}</div></div>`).join("") || "No section alerts up to this week."}</div>`;
}

// ------------------------------------------------------------------ evaluation
async function loadEvaluation() {
  let r;
  try { r = await api("/api/evaluation"); } catch (e) { $("#view-evaluation").innerHTML = esc(e.message); return; }
  const m = r.metrics, d = r.dataset;
  $("#view-evaluation").innerHTML = `<p class="lead">Produced by <code>python -m learnpulse build</code> on a synthetic school (seed ${d.seed}): ${d.students.toLocaleString()} students, ${d.erp_rows_total.toLocaleString()} ERP rows, ${d.trajectory_students} students with an injected trajectory. These numbers show that the pipeline works end to end; they are not results from a real school.</p>
    <div class="summary">${stat("PR-AUC model", m.pr_auc_model)}${stat("PR-AUC rules only", m.pr_auc_rules_only)}${stat("PR-AUC attendance < 75%", m.pr_auc_attendance_below_75)}${stat("Recall within 4 weeks", Math.round(m.recall_listed_within_4w * 100) + "%")}${stat("Median lead time", m.lead_time_weeks_median + " weeks")}</div>
    <table class="table"><thead><tr><th>Weekly list (test weeks ${r.model.test_weeks.join("-")})</th><th>Listed</th><th>On an injected trajectory</th><th>Declined within 6 weeks</th></tr></thead><tbody>
    <tr><td>Priority = 0.6 model + 0.4 rules</td><td>${m.list_priority_listed}</td><td>${m.list_priority_precision_trajectory}</td><td>${m.list_priority_precision_label}</td></tr>
    <tr><td>Rules only</td><td>${m.list_rules_only_listed}</td><td>${m.list_rules_only_precision_trajectory}</td><td>${m.list_rules_only_precision_label}</td></tr></tbody></table>`;
}

init();
