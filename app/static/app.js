import WaveSurfer from "./vendor/wavesurfer-7.12.12/wavesurfer.esm.js";
import RecordPlugin from "./vendor/wavesurfer-7.12.12/record.esm.js";

const $ = (id) => document.getElementById(id);
let blob = null;

fetch("/health").then(r => r.json()).then(h => { $("stub").hidden = !h.stub; }).catch(() => {});

// Un solo WaveSurfer: muestra la onda en vivo al grabar y la onda del audio cargado al reproducir.
const css = getComputedStyle(document.documentElement);
const ws = WaveSurfer.create({
  container: "#wave", height: 56, barWidth: 2, barGap: 1, barRadius: 2, cursorWidth: 1,
  waveColor: css.getPropertyValue("--wave").trim(), progressColor: css.getPropertyValue("--acc").trim(),
  cursorColor: css.getPropertyValue("--fg").trim(),
});
const rec = ws.registerPlugin(RecordPlugin.create({ renderRecordedAudio: false, continuousWaveform: true }));

const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const showTime = () => { $("time").textContent = `${fmt(ws.getCurrentTime())} / ${fmt(ws.getDuration())}`; };
ws.on("ready", showTime);
ws.on("timeupdate", showTime);
ws.on("play", () => { $("play").textContent = "❚❚"; $("play").ariaLabel = "Pausar"; });
ws.on("pause", () => { $("play").textContent = "▶"; $("play").ariaLabel = "Reproducir"; });
ws.on("error", () => showError("El navegador no pudo mostrar este audio; igual se puede analizar."));
$("play").onclick = () => ws.playPause();

function setBlob(b, name) {
  blob = b; blob.fname = name;
  ws.loadBlob(b);
  $("play").disabled = false; $("send").disabled = false;
  $("result").hidden = true; $("error").hidden = true;
}

$("file").onchange = (e) => { const f = e.target.files[0]; if (f) setBlob(f, f.name); };

rec.on("record-progress", (ms) => { $("time").textContent = fmt(ms / 1000); });
rec.on("record-end", (b) => {
  $("rec").textContent = "● Grabar"; $("rec").classList.remove("recording"); $("file").disabled = false;
  const ext = b.type.includes("mp4") ? ".mp4" : b.type.includes("ogg") ? ".ogg" : ".webm";
  setBlob(b, "grabacion" + ext);
});

$("rec").onclick = async () => {
  if (rec.isRecording()) { rec.stopRecording(); return; }
  ws.pause(); ws.empty();
  $("play").disabled = true; $("send").disabled = true; $("result").hidden = true; $("error").hidden = true;
  try {
    await rec.startRecording();
    $("rec").textContent = "■ Detener"; $("rec").classList.add("recording"); $("file").disabled = true;
  } catch (err) { showError("No se pudo acceder al micrófono: " + err.message); }
};

const NEAR = 0.05;   // |puntaje - umbral| por debajo de esto se marca "cerca del umbral"

// Una sola escala con el puntaje del modelo (salida sigmoid, no una probabilidad calibrada) y el
// umbral marcado: la etiqueta depende de en que lado del umbral cae el puntaje, no de si supera 0.5.
function renderResult(j) {
  const [neg, pos] = Object.keys(j.probabilities);          // ["HC", "AD"]: indice 1 = clase positiva
  const s = j.probabilities[pos], thr = j.threshold, d = s - thr;
  const pct = (x) => (x * 100).toFixed(1) + "%";
  const near = Math.abs(d) < NEAR;
  const side = d >= 0 ? "por encima" : "por debajo";
  return `<strong>Resultado del modelo: ${j.label}</strong>${near ? ' <span class="near">cerca del umbral</span>' : ""}
    <div class="scale" role="img" aria-label="Puntaje ${s.toFixed(2)}, umbral ${thr.toFixed(2)}">
      <div class="zone neg" style="width:${pct(thr)}">${neg}</div><div class="zone pos">${pos}</div>
      <div class="thr" style="left:${pct(thr)}"></div>
      <div class="score" style="left:${pct(s)}"></div>
    </div>
    <div class="ticks"><span>0</span><span class="thr-label" style="left:${pct(thr)}">umbral ${thr.toFixed(2)}</span><span>1</span></div>
    <p>Puntaje del modelo: <b>${s.toFixed(2)}</b>, ${Math.abs(d).toFixed(2)} ${side} del umbral → ${j.label}.</p>
    <small>El puntaje es la salida del modelo entre 0 y 1; no es la probabilidad de tener Alzheimer. El umbral
      (${thr.toFixed(2)}) se eligió durante el entrenamiento junto con este modelo: un puntaje de ${pos} se informa
      solo si llega a ese valor.</small>`;
}

function showError(msg) { $("error").textContent = msg; $("error").hidden = false; }

$("send").onclick = async () => {
  $("send").disabled = true; $("send").textContent = "Analizando… (puede tardar)"; $("error").hidden = true;
  try {
    const fd = new FormData(); fd.append("file", blob, blob.fname);
    const r = await fetch("/predict", { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || r.statusText);
    $("result").innerHTML = renderResult(j);
    $("result").hidden = false;
  } catch (e) { showError(e.message); }
  finally { $("send").disabled = false; $("send").textContent = "Analizar"; }
};
