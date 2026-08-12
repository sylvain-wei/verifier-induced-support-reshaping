(() => {
  "use strict";

  document.documentElement.classList.add("js");

  const navToggle = document.querySelector("[data-nav-toggle]");
  const navLinksPanel = document.querySelector("[data-nav-links]");

  if (navToggle && navLinksPanel) {
    const toggleSymbol = navToggle.querySelector("span[aria-hidden]");
    navToggle.addEventListener("click", () => {
      const isOpen = navToggle.getAttribute("aria-expanded") === "true";
      navToggle.setAttribute("aria-expanded", String(!isOpen));
      navLinksPanel.classList.toggle("is-open", !isOpen);
      if (toggleSymbol) toggleSymbol.textContent = isOpen ? "+" : "−";
    });

    navLinksPanel.querySelectorAll("a").forEach((link) => {
      link.addEventListener("click", () => {
        navToggle.setAttribute("aria-expanded", "false");
        navLinksPanel.classList.remove("is-open");
        if (toggleSymbol) toggleSymbol.textContent = "+";
      });
    });
  }

  const progressBar = document.querySelector("[data-reading-progress]");
  if (progressBar) {
    let progressFrame = null;
    const updateProgress = () => {
      const scrollable = document.documentElement.scrollHeight - window.innerHeight;
      const progress = scrollable > 0 ? Math.min(1, window.scrollY / scrollable) : 0;
      progressBar.style.width = `${progress * 100}%`;
      progressFrame = null;
    };
    const requestProgress = () => {
      if (progressFrame === null) progressFrame = window.requestAnimationFrame(updateProgress);
    };
    updateProgress();
    window.addEventListener("scroll", requestProgress, { passive: true });
    window.addEventListener("resize", requestProgress);
  }

  const dialog = document.querySelector("[data-figure-dialog]");
  const dialogImage = dialog?.querySelector("[data-dialog-image]");
  const dialogCaption = dialog?.querySelector("[data-dialog-caption]");
  const dialogClose = dialog?.querySelector("[data-dialog-close]");
  let returnFocus = null;

  if (dialog && dialogImage && dialogCaption && typeof dialog.showModal === "function") {
    document.querySelectorAll("[data-zoom]").forEach((link) => {
      link.addEventListener("click", (event) => {
        event.preventDefault();
        returnFocus = link;
        dialogImage.src = link.getAttribute("href");
        dialogImage.alt = link.dataset.alt || "Expanded paper figure";
        dialogCaption.textContent = link.dataset.caption || "Expanded paper figure";
        dialog.showModal();
        dialogClose.focus();
      });
    });

    const closeDialog = () => dialog.close();
    dialogClose.addEventListener("click", closeDialog);
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) closeDialog();
    });
    dialog.addEventListener("close", () => {
      dialogImage.removeAttribute("src");
      returnFocus?.focus();
      returnFocus = null;
    });
  }

  const copyButton = document.querySelector("[data-copy-citation]");
  const copyStatus = document.querySelector("[data-copy-status]");
  const bibtex = document.querySelector("#bibtex");

  if (copyButton && copyStatus && bibtex) {
    copyButton.addEventListener("click", async () => {
      const citation = bibtex.textContent.trim();
      try {
        await navigator.clipboard.writeText(citation);
        copyStatus.textContent = "BibTeX copied to clipboard.";
      } catch {
        const range = document.createRange();
        const selection = window.getSelection();
        range.selectNodeContents(bibtex);
        selection.removeAllRanges();
        selection.addRange(range);
        copyStatus.textContent = "BibTeX selected. Press Ctrl+C or Command+C to copy.";
      }
    });
  }

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const revealTargets = document.querySelectorAll(
    ".scientific-figure, .result-ledger, .metric-definitions > div, .preservation-list article"
  );

  if (!reducedMotion && "IntersectionObserver" in window) {
    revealTargets.forEach((target) => target.setAttribute("data-reveal", ""));
    const revealObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    }, { rootMargin: "0px 0px -8%", threshold: 0.08 });
    revealTargets.forEach((target) => revealObserver.observe(target));
  }

  const atlas = document.querySelector(".support-atlas");
  if (atlas && !reducedMotion && "IntersectionObserver" in window) {
    const atlasObserver = new IntersectionObserver((entries, observer) => {
      if (!entries[0]?.isIntersecting) return;
      atlas.classList.add("is-active");
      observer.disconnect();
    }, { threshold: 0.3 });
    atlasObserver.observe(atlas);
  }

  const navLinks = [...document.querySelectorAll(".nav-links a[href^='#']")];
  const sections = navLinks
    .map((link) => document.querySelector(link.getAttribute("href")))
    .filter(Boolean);

  if ("IntersectionObserver" in window && sections.length) {
    const observer = new IntersectionObserver((entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      navLinks.forEach((link) => {
        const active = link.getAttribute("href") === `#${visible.target.id}`;
        if (active) link.setAttribute("aria-current", "location");
        else link.removeAttribute("aria-current");
      });
    }, { rootMargin: "-20% 0px -65%", threshold: [0, 0.2, 0.5] });
    sections.forEach((section) => observer.observe(section));
  }
})();
