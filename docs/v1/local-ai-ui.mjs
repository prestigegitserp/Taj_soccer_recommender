export function mountLocalAI(getEvidence) {
  const byId = (id) => document.getElementById(id);
  const start = byId("aiStart"), stop = byId("aiStop");
  const ask = byId("aiAsk"), input = byId("aiQuestion");
  const status = byId("aiStatus"), output = byId("aiAnswer");
  const progress = byId("aiProgress");
  let worker = null, ready = false;
  function shutdown() {
    if (worker) worker.terminate();
    worker = null; ready = false; start.disabled = false;
    stop.disabled = true; ask.disabled = true; input.disabled = true;
    progress.hidden = true; status.textContent = "مدل متوقف شد.";
  }
  stop.addEventListener("click", shutdown);
  start.addEventListener("click", async () => {
    if (worker) return;
    const gpu = navigator.gpu;
    if (!gpu || !(await gpu.requestAdapter())) {
      status.textContent = "WebGPU قابل استفاده نیست؛ مدل آماری همچنان فعال است.";
      return;
    }
    worker = new Worker(new URL("./ai-worker.mjs", import.meta.url), {type:"module"});
    start.disabled = true; stop.disabled = false; progress.hidden = false;
    worker.onmessage = ({data}) => {
      if (data.type === "progress") {
        progress.value = Math.round((data.value || 0)*100);
        status.textContent = data.text;
      } else if (data.type === "ready") {
        ready = true; progress.hidden = true;
        input.disabled = false; ask.disabled = false;
        status.textContent = "مدل زبانی روی دستگاه آماده شد.";
      } else if (data.type === "answer") {
        output.textContent = data.text; ask.disabled = false;
      } else if (data.type === "error") {
        output.textContent = data.message; ask.disabled = !ready;
      }
    };
    worker.onerror = () => {shutdown(); status.textContent = "راه‌اندازی مدل ناموفق بود.";};
    worker.postMessage({type:"init"});
  });
  ask.addEventListener("click", async () => {
    if (!ready || !worker) return;
    const snapshot = await getEvidence();
    if (!snapshot) return;
    ask.disabled = true; output.textContent = "در حال تحلیل...";
    worker.postMessage({type:"generate", snapshot, question:input.value});
  });
}
