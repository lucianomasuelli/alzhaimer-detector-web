const $ = (id) => document.getElementById(id);
let blob = null, recorder = null, chunks = [], t0 = 0, tick = null;

fetch("/health").then(r => r.json()).then(h => { $("stub").hidden = !h.stub; }).catch(() => {});

function setBlob(b, name) {
  blob = b; blob.fname = name;
  $("player").src = URL.createObjectURL(b); $("player").hidden = false;
  $("send").disabled = false; $("result").hidden = true; $("error").hidden = true;
}

$("file").onchange = (e) => { const f = e.target.files[0]; if (f) setBlob(f, f.name); };

$("rec").onclick = async () => {
  if (recorder && recorder.state === "recording") { recorder.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new MediaRecorder(stream); chunks = [];
    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = () => {
      stream.getTracks().forEach(t => t.stop()); clearInterval(tick);
      $("rec").textContent = "● Grabar"; $("rec").classList.remove("recording"); $("timer").textContent = "";
      const ext = recorder.mimeType.includes("mp4") ? ".mp4" : recorder.mimeType.includes("ogg") ? ".ogg" : ".webm";
      setBlob(new Blob(chunks, { type: recorder.mimeType }), "grabacion" + ext);
    };
    recorder.start(); t0 = Date.now();
    $("rec").textContent = "■ Detener"; $("rec").classList.add("recording");
    tick = setInterval(() => { $("timer").textContent = ((Date.now() - t0) / 1000).toFixed(0) + " s"; }, 500);
  } catch (err) { showError("No se pudo acceder al micrófono: " + err.message); }
};

function showError(msg) { $("error").textContent = msg; $("error").hidden = false; }

$("send").onclick = async () => {
  $("send").disabled = true; $("send").textContent = "Analizando… (puede tardar)"; $("error").hidden = true;
  try {
    const fd = new FormData(); fd.append("file", blob, blob.fname);
    const r = await fetch("/predict", { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    const rows = Object.entries(j.probabilities).map(([k, v]) =>
      `<div>${k}: ${(v * 100).toFixed(1)}%</div><div class="bar"><div style="width:${v * 100}%"></div></div>`).join("");
    $("result").innerHTML = `<strong>Resultado del modelo: ${j.label}</strong>${rows}
      <small>Umbral ${j.threshold}; duración ${j.duration_s.toFixed(1)} s. Salida de un prototipo experimental: no es un diagnóstico ni una medida de riesgo.</small>`;
    $("result").hidden = false;
  } catch (e) { showError(e.message); }
  finally { $("send").disabled = false; $("send").textContent = "Analizar"; }
};
