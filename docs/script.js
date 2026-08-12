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

  const fragmentStatus = document.querySelector("[data-fragment-status]");
  const paperFragments = document.querySelectorAll(".paper-fragment");
  const fragmentAnimations = new WeakMap();

  const seededUnit = (seed) => {
    const value = Math.sin(seed * 12.9898 + 78.233) * 43758.5453;
    return value - Math.floor(value);
  };

  const prepareFragment = (fragment, fragmentIndex) => {
    const textNode = fragment.querySelector("p");
    if (!textNode) return;

    const originalText = textNode.textContent.trim();
    const source = fragment.dataset.fragmentSource || "Paper excerpt";
    textNode.textContent = "";
    textNode.setAttribute("aria-hidden", "true");

    [...originalText].forEach((character, charIndex) => {
      const span = document.createElement("span");
      span.className = "paper-fragment__char";
      span.textContent = character === " " ? "\u00a0" : character;
      span.dataset.charIndex = String(charIndex);
      textNode.append(span);
    });

    fragment.setAttribute("role", "button");
    fragment.setAttribute("tabindex", "0");
    fragment.setAttribute("aria-pressed", "false");
    fragment.setAttribute("aria-label", `${source}: ${originalText} Activate to scatter this excerpt.`);
    fragment.dataset.fragmentIndex = String(fragmentIndex);
    fragment.dataset.fragmentText = originalText;
  };

  const scatterFragment = (fragment, impact) => {
    if (fragment.classList.contains("is-busy")) return;
    const characters = [...fragment.querySelectorAll(".paper-fragment__char")];
    const rect = fragment.getBoundingClientRect();
    const impactX = impact?.x ?? rect.left + rect.width / 2;
    const impactY = impact?.y ?? rect.top + rect.height / 2;
    const inward = fragment.classList.contains("paper-fragment--left") ? 1 : -1;
    const fragmentSeed = Number(fragment.dataset.fragmentIndex || 0) * 997;
    const animations = [];

    fragment.classList.add("is-shattered", "is-busy");
    fragment.setAttribute("aria-pressed", "true");

    characters.forEach((character, index) => {
      const charRect = character.getBoundingClientRect();
      const charX = charRect.left + charRect.width / 2;
      const charY = charRect.top + charRect.height / 2;
      const horizontal = (charX - impactX) / Math.max(rect.width, 1);
      const vertical = (charY - impactY) / Math.max(rect.height, 1);
      const randomA = seededUnit(fragmentSeed + index * 3 + 1);
      const randomB = seededUnit(fragmentSeed + index * 3 + 2);
      const randomC = seededUnit(fragmentSeed + index * 3 + 3);
      const dx = inward * (28 + randomA * 118) + horizontal * 92;
      const lift = -18 - randomB * 58 + vertical * 18;
      const dy = 72 + randomC * 230 + Math.abs(vertical) * 55;
      const rotation = (randomA - 0.5) * 330 + inward * 22;
      const duration = 760 + randomB * 640;
      const delay = randomC * 115;

      const animation = character.animate([
        { transform: "translate(0, 0) rotate(0deg)", opacity: 1, offset: 0 },
        { transform: `translate(${dx * 0.16}px, ${lift}px) rotate(${rotation * 0.12}deg)`, opacity: 1, offset: 0.16 },
        { transform: `translate(${dx}px, ${dy}px) rotate(${rotation}deg)`, opacity: 0.34, offset: 1 }
      ], {
        duration,
        delay,
        easing: "cubic-bezier(0.18, 0.72, 0.22, 1)",
        fill: "forwards"
      });
      animations.push(animation);
    });

    fragmentAnimations.set(fragment, animations);
    window.setTimeout(() => fragment.classList.remove("is-busy"), 1550);
    if (fragmentStatus) fragmentStatus.textContent = `${fragment.dataset.fragmentSource} excerpt scattered. Activate it again to recompose.`;
  };

  const recomposeFragment = (fragment) => {
    if (fragment.classList.contains("is-busy")) return;
    const characters = [...fragment.querySelectorAll(".paper-fragment__char")];
    const previousAnimations = fragmentAnimations.get(fragment) || [];
    const fragmentSeed = Number(fragment.dataset.fragmentIndex || 0) * 613;
    const restores = [];

    fragment.classList.add("is-busy");
    characters.forEach((character, index) => {
      const computed = window.getComputedStyle(character);
      const startTransform = computed.transform === "none" ? "translate(0, 0)" : computed.transform;
      const startOpacity = computed.opacity;
      previousAnimations[index]?.cancel();
      const animation = character.animate([
        { transform: startTransform, opacity: startOpacity },
        { transform: "translate(0, 0) rotate(0deg)", opacity: 1 }
      ], {
        duration: 470 + seededUnit(fragmentSeed + index) * 330,
        delay: seededUnit(fragmentSeed + index * 2) * 90,
        easing: "cubic-bezier(0.22, 0.72, 0.24, 1)",
        fill: "forwards"
      });
      restores.push(animation.finished.catch(() => undefined));
    });

    let finalized = false;
    const finishRecompose = () => {
      if (finalized) return;
      finalized = true;
      characters.forEach((character) => character.getAnimations().forEach((animation) => animation.cancel()));
      fragment.classList.remove("is-shattered", "is-busy");
      fragment.setAttribute("aria-pressed", "false");
      fragmentAnimations.delete(fragment);
    };
    Promise.all(restores).then(finishRecompose);
    window.setTimeout(finishRecompose, 1100);
    if (fragmentStatus) fragmentStatus.textContent = `${fragment.dataset.fragmentSource} excerpt recomposed.`;
  };

  paperFragments.forEach((fragment, index) => {
    prepareFragment(fragment, index);
    const activate = (impact) => {
      if (reducedMotion) {
        const scattered = fragment.classList.toggle("is-shattered");
        fragment.setAttribute("aria-pressed", String(scattered));
        if (fragmentStatus) fragmentStatus.textContent = scattered ? "Excerpt de-emphasized without animation." : "Excerpt restored.";
        return;
      }
      if (fragment.classList.contains("is-shattered")) recomposeFragment(fragment);
      else scatterFragment(fragment, impact);
    };

    fragment.addEventListener("click", (event) => activate({ x: event.clientX, y: event.clientY }));
    fragment.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      activate();
    });
  });

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
