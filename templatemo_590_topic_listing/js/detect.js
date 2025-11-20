(function () {
  const API_BASE = "https://bharatfact-1.onrender.com";
  // Optional threshold override via page URL: ?th=0.65
  const urlParams = new URLSearchParams(window.location.search);
  const TH_PARAM = urlParams.get("th");
  const form = document.getElementById("detect-form");
  const input = document.getElementById("keyword");

  // Modal elements
  const modalEl = document.getElementById("detectModal");
  const modal = modalEl ? new bootstrap.Modal(modalEl) : null;
  const labelEl = document.getElementById("detectModalLabelText");
  const confEl = document.getElementById("detectModalConfPct");
  const barEl = document.getElementById("detectModalConfBar");
  const langEl = document.getElementById("detectModalLang");
  const loading = document.getElementById("detectModalLoading");
  const errorBox = document.getElementById("detectModalError");

  function detectLang(t) {
    return /[\u0900-\u097F]/.test(t) ? "Hindi" : "English";
  }

  function setError(msg) {
    if (!errorBox) return;
    if (!msg) {
      errorBox.style.display = "none";
      errorBox.textContent = "";
      return;
    }
    errorBox.style.display = "block";
    errorBox.textContent = msg;
  }
  function setLoading(v) {
    if (loading) loading.style.display = v ? "block" : "none";
  }

  async function predict(text) {
    const endpoint = API_BASE + "/predict" + (TH_PARAM ? ("?th=" + encodeURIComponent(TH_PARAM)) : "");
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) {
      let detail = "";
      try {
        detail = await res.text();
      } catch (e) {}
      throw new Error(detail || "HTTP " + res.status);
    }
    return await res.json();
  }

  form.addEventListener("submit", async function (e) {
    // If form has action attribute, allow normal submission (redirect)
    if (
      form.action &&
      form.action !== "#" &&
      form.action !== window.location.href
    ) {
      const text = (input.value || "").trim();
      if (!text) {
        e.preventDefault();
        alert("Please enter some news text to analyze.");
        return;
      }
      // Allow form to submit normally (redirect to predict.html)
      return;
    }

    // Otherwise, use modal for inline results
    e.preventDefault();
    setError("");
    if (modal) modal.show();
    const text = (input.value || "").trim();
    if (!text) {
      setError("Please paste some news text.");
      return;
    }
    const lang = detectLang(text);
    if (langEl) langEl.textContent = lang;
    setLoading(true);
    try {
      const out = await predict(text);
      if (labelEl) labelEl.textContent = out.label;
      const pct = Math.round((out.confidence || 0) * 100);
      if (confEl) confEl.textContent = pct + "%";
      if (barEl) {
        barEl.style.width = pct + "%";
        barEl.className =
          "progress-bar " +
          (String(out.label).toLowerCase().includes("fake")
            ? "bg-danger"
            : "bg-success");
      }
    } catch (err) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  });
})();
