(() => {
  "use strict";

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
