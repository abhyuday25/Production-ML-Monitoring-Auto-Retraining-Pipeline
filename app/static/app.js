const form = document.querySelector("#upload-form");
const fileInput = document.querySelector("#dataset-file");
const fileLabel = document.querySelector("#file-label");
const uploadButton = document.querySelector("#upload-button");
const healthDot = document.querySelector("#health-dot");
const healthText = document.querySelector("#health-text");
const jobStatus = document.querySelector("#job-status");
const jobMessage = document.querySelector("#job-message");
const resultJson = document.querySelector("#result-json");
const metricAccuracy = document.querySelector("#metric-accuracy");
const metricF1 = document.querySelector("#metric-f1");
const metricRoc = document.querySelector("#metric-roc");
const metricVersion = document.querySelector("#metric-version");

let activeJobId = null;
let pollTimer = null;

fileInput.addEventListener("change", () => {
  fileLabel.textContent = fileInput.files.length ? fileInput.files[0].name : "Choose dataset";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!fileInput.files.length) {
    setJob("waiting", "Choose a CSV file first.");
    return;
  }

  const payload = new FormData();
  payload.append("file", fileInput.files[0]);
  uploadButton.disabled = true;
  setJob("uploading", "Uploading dataset.");

  try {
    const response = await fetch("/training/upload", {
      method: "POST",
      body: payload,
    });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.detail || "Upload failed.");
    }
    activeJobId = body.id;
    renderJob(body);
    startPolling();
  } catch (error) {
    setJob("failed", error.message);
    uploadButton.disabled = false;
  }
});

async function refreshHealth() {
  try {
    const response = await fetch("/health");
    const body = await response.json();
    healthDot.className = `status-dot ${body.model_loaded ? "ok" : "bad"}`;
    healthText.textContent = body.model_loaded ? "API ready" : "Model unavailable";
  } catch {
    healthDot.className = "status-dot bad";
    healthText.textContent = "API offline";
  }
}

async function loadLatestJob() {
  try {
    const response = await fetch("/training/jobs");
    const body = await response.json();
    if (!body.jobs.length) {
      return;
    }
    renderJob(body.jobs[0]);
    if (["queued", "running"].includes(body.jobs[0].status)) {
      activeJobId = body.jobs[0].id;
      startPolling();
    }
  } catch {
    setJob("unavailable", "Training job history is unavailable.");
  }
}

function startPolling() {
  window.clearInterval(pollTimer);
  pollTimer = window.setInterval(refreshActiveJob, 2000);
  refreshActiveJob();
}

async function refreshActiveJob() {
  if (!activeJobId) {
    return;
  }
  const response = await fetch(`/training/jobs/${activeJobId}`);
  const job = await response.json();
  renderJob(job);
  if (!["queued", "running"].includes(job.status)) {
    window.clearInterval(pollTimer);
    uploadButton.disabled = false;
  }
}

function renderJob(job) {
  setJob(job.status, job.error || job.message);
  if (job.result) {
    resultJson.textContent = JSON.stringify(job.result, null, 2);
    const metrics = job.result.metrics || {};
    metricAccuracy.textContent = formatMetric(metrics.accuracy);
    metricF1.textContent = formatMetric(metrics.f1);
    metricRoc.textContent = formatMetric(metrics.roc_auc);
    metricVersion.textContent = job.result.model_version || "--";
  }
}

function setJob(status, message) {
  jobStatus.textContent = status;
  jobMessage.textContent = message;
}

function formatMetric(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) {
    return "--";
  }
  return Number(value).toFixed(3);
}

refreshHealth();
loadLatestJob();
window.setInterval(refreshHealth, 5000);
