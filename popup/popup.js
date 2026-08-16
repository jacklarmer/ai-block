// LocalLens popup controller
const els = {
  engine: document.getElementById("engineState"),
  runtime: document.getElementById("runtimeInfo"),
  modelVersion: document.getElementById("modelVersion"),
  enabled: document.getElementById("enabled"),
  blockFlagged: document.getElementById("blockFlagged"),
  threshold: document.getElementById("threshold"),
  thresholdVal: document.getElementById("thresholdVal"),
  rescan: document.getElementById("rescanBtn"),
  lastScan: document.getElementById("lastScan"),
  statImages: document.getElementById("statImages"),
  statAI: document.getElementById("statAI"),
  statReal: document.getElementById("statReal"),
};

async function refresh() {
  // query background for model state
  try {
    const state = await chrome.runtime.sendMessage({ type: "locallens:state" });
    if (state && state.ready) {
      setPill("ok", "model ready · offline");
    } else {
      setPill("warn", "model not ready");
    }
    els.runtime.textContent = state && state.bundled ? "bundled weights" : "runtime: WebGPU/WASM";
    if (els.modelVersion && state && state.version) {
      els.modelVersion.textContent = "model " + state.version;
    }
  } catch (e) {
    setPill("err", "extension error");
  }

  // storage settings
  chrome.storage.local.get(["locallens_enabled", "locallens_threshold", "locallens_block"], (s) => {
    if (typeof s.locallens_enabled === "boolean") els.enabled.checked = s.locallens_enabled;
    els.blockFlagged.checked = s.locallens_block === true;
    if (typeof s.locallens_threshold === "number") {
      els.threshold.value = Math.round(s.locallens_threshold * 100);
      els.thresholdVal.textContent = els.threshold.value + "%";
    }
  });
}

function setPill(kind, text) {
  els.engine.className = "pill " + kind;
  els.engine.textContent = text;
}

els.enabled.addEventListener("change", () => {
  chrome.storage.local.set({ locallens_enabled: els.enabled.checked });
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs[0] && tabs[0].id) {
      chrome.tabs.sendMessage(tabs[0].id, {
        type: els.enabled.checked ? "locallens:enable" : "locallens:disable",
      }).catch(() => {});
    }
  });
});

els.blockFlagged.addEventListener("change", () => {
  const on = els.blockFlagged.checked;
  chrome.storage.local.set({ locallens_block: on });
  // push to the active page immediately so it takes effect without a reload
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs[0] && tabs[0].id) {
      chrome.tabs.sendMessage(tabs[0].id, { type: "locallens:block", value: on }).catch(() => {});
    }
  });
});

els.threshold.addEventListener("input", () => {
  const v = parseInt(els.threshold.value, 10) / 100;
  els.thresholdVal.textContent = els.threshold.value + "%";
  chrome.storage.local.set({ locallens_threshold: v });
  // push the new threshold to the active page immediately so it takes effect
  // without a reload
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs[0] && tabs[0].id) {
      chrome.tabs.sendMessage(tabs[0].id, { type: "locallens:threshold", value: v }).catch(() => {});
    }
  });
});

els.rescan.addEventListener("click", () => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs[0] && tabs[0].id) {
      chrome.tabs.sendMessage(tabs[0].id, { type: "locallens:scan" }).then(
        () => {
          els.lastScan.textContent = "scan triggered ✓";
        },
        () => {
          els.lastScan.textContent = "no page to scan";
        }
      );
    }
  });
});

refresh();
