/* Adlane Ladjal — site scripts (vanilla JS, no framework) */
(function () {
  'use strict';

  const root = document.documentElement;

  /* ── Theme toggle ─────────────────────────────────────────────────────── */
  const stored = localStorage.getItem('portfolio-theme');
  const initialTheme = stored || (root.dataset.defaultTheme || 'dark');
  root.setAttribute('data-theme', initialTheme);

  document.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-theme-toggle]');
    if (!btn) return;
    const next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    localStorage.setItem('portfolio-theme', next);
  });

  /* ── Custom cursor (only on hover-capable, fine-pointer devices) ──────── */
  const supportsCursor = window.matchMedia('(hover: hover) and (pointer: fine)').matches;
  if (supportsCursor) {
    const dot = document.createElement('div');
    const ring = document.createElement('div');
    dot.className = 'cursor-dot';
    ring.className = 'cursor-ring';
    document.body.appendChild(dot);
    document.body.appendChild(ring);

    let pos = { x: -100, y: -100 };
    let ringPos = { x: -100, y: -100 };

    const onMove = (e) => {
      pos = { x: e.clientX, y: e.clientY };
      dot.style.transform = `translate(${e.clientX}px, ${e.clientY}px) translate(-50%, -50%)`;
    };
    const lerp = (a, b, t) => a + (b - a) * t;
    const tick = () => {
      ringPos.x = lerp(ringPos.x, pos.x, 0.12);
      ringPos.y = lerp(ringPos.y, pos.y, 0.12);
      ring.style.transform = `translate(${ringPos.x}px, ${ringPos.y}px) translate(-50%, -50%)`;
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);

    window.addEventListener('mousemove', onMove, { passive: true });
    document.addEventListener('mouseover', (e) => {
      if (e.target.closest('a, button, [data-hover]')) {
        ring.classList.add('cursor-ring--hover');
        dot.classList.add('cursor-dot--hover');
      }
    });
    document.addEventListener('mouseout', () => {
      ring.classList.remove('cursor-ring--hover');
      dot.classList.remove('cursor-dot--hover');
    });
  }

  /* ── Navbar scroll state + smooth scroll + mobile menu ────────────────── */
  const navbar = document.querySelector('.navbar');
  if (navbar) {
    const onScroll = () => {
      if (window.scrollY > 40) navbar.classList.add('navbar--scrolled');
      else navbar.classList.remove('navbar--scrolled');
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();

    const hamburger = navbar.querySelector('.hamburger');
    if (hamburger) {
      hamburger.addEventListener('click', () => navbar.classList.toggle('navbar--open'));
    }

    navbar.querySelectorAll('[data-scroll-to]').forEach((el) => {
      el.addEventListener('click', (e) => {
        const id = el.getAttribute('data-scroll-to');
        const target = document.getElementById(id);
        if (target) {
          e.preventDefault();
          target.scrollIntoView({ behavior: 'smooth', block: 'start' });
          navbar.classList.remove('navbar--open');
        }
      });
    });

    const brand = navbar.querySelector('.navbar__brand');
    if (brand) {
      brand.addEventListener('click', (e) => {
        e.preventDefault();
        window.scrollTo({ top: 0, behavior: 'smooth' });
      });
    }
  }

  /* ── Reveal on scroll ──────────────────────────────────────────────────── */
  const revealObs = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('reveal--visible');
        revealObs.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });
  document.querySelectorAll('.reveal').forEach((el) => revealObs.observe(el));

  const sectionObs = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('section--visible');
        sectionObs.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12 });
  document.querySelectorAll('.section').forEach((el) => sectionObs.observe(el));

  /* ── Hero typewriter ──────────────────────────────────────────────────── */
  const taglineEl = document.querySelector('[data-typewriter]');
  if (taglineEl) {
    const lines = (taglineEl.dataset.lines || '').split('|').filter(Boolean);
    if (lines.length) {
      taglineEl.innerHTML = '';
      const cursor = document.createElement('span');
      cursor.className = 'hero__cursor hero__cursor--on';
      cursor.textContent = '|';

      let lineIdx = 0;
      let charIdx = 0;
      let currentLineEl = null;

      const startLine = () => {
        currentLineEl = document.createElement('div');
        currentLineEl.className = 'hero__tagline-line hero__tagline-line--current';
        if (taglineEl.contains(cursor)) cursor.remove();
        currentLineEl.appendChild(document.createTextNode(''));
        currentLineEl.appendChild(cursor);
        taglineEl.appendChild(currentLineEl);
      };

      const tick = () => {
        const txt = lines[lineIdx];
        if (charIdx < txt.length) {
          charIdx++;
          currentLineEl.firstChild.nodeValue = txt.slice(0, charIdx);
          setTimeout(tick, 55);
        } else if (lineIdx < lines.length - 1) {
          // Finalize current line and start next
          currentLineEl.classList.remove('hero__tagline-line--current');
          if (currentLineEl.contains(cursor)) cursor.remove();
          lineIdx++;
          charIdx = 0;
          startLine();
          setTimeout(tick, 600);
        } else {
          // Done — blink cursor
          setInterval(() => cursor.classList.toggle('hero__cursor--on'), 530);
        }
      };

      startLine();
      setTimeout(tick, 400);
    }
  }

  /* ── Hero parallax ─────────────────────────────────────────────────────── */
  const parallax = document.querySelector('[data-parallax]');
  if (parallax) {
    const onScroll = () => {
      const y = window.scrollY;
      parallax.style.transform = `translateY(${y * 0.35}px)`;
    };
    window.addEventListener('scroll', onScroll, { passive: true });
  }

  /* ── Timeline accordion ───────────────────────────────────────────────── */
  document.querySelectorAll('.timeline').forEach((tl) => {
    const items = tl.querySelectorAll('.timeline__item');
    const refreshChevrons = () => {
      items.forEach((it) => {
        const ch = it.querySelector('.timeline__chevron');
        if (ch) ch.textContent = it.classList.contains('timeline__item--open') ? '−' : '+';
      });
    };
    items.forEach((item, i) => {
      if (i === 0) item.classList.add('timeline__item--open');
      item.addEventListener('click', (e) => {
        if (e.target.closest('a')) return;
        const isOpen = item.classList.contains('timeline__item--open');
        items.forEach((it) => it.classList.remove('timeline__item--open'));
        if (!isOpen) item.classList.add('timeline__item--open');
        refreshChevrons();
      });
    });
    refreshChevrons();
  });

  /* ── Contact form (Formspree) ─────────────────────────────────────────── */
  const form = document.querySelector('[data-contact-form]');
  if (form) {
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const endpoint = form.getAttribute('action');
      const submitBtn = form.querySelector('button[type="submit"]');
      const errEl = form.querySelector('.form-error');
      if (errEl) errEl.style.display = 'none';
      const originalText = submitBtn ? submitBtn.textContent : '';
      if (submitBtn) { submitBtn.disabled = true; submitBtn.textContent = 'Sending…'; }

      try {
        const data = new FormData(form);
        const res = await fetch(endpoint, {
          method: 'POST',
          headers: { Accept: 'application/json' },
          body: data,
        });
        if (res.ok) {
          form.reset();
          form.style.display = 'none';
          const success = document.querySelector('.contact__success');
          if (success) success.classList.remove('contact__success--hidden');
        } else {
          if (errEl) { errEl.style.display = 'block'; }
        }
      } catch {
        if (errEl) errEl.style.display = 'block';
      } finally {
        if (submitBtn) { submitBtn.disabled = false; submitBtn.textContent = originalText; }
      }
    });
  }

  /* ── Article: scroll progress, code copy, TOC active state ────────────── */
  const progress = document.querySelector('.progress-bar');
  if (progress) {
    const onScroll = () => {
      const s = document.documentElement;
      const pct = s.scrollTop / Math.max(1, s.scrollHeight - s.clientHeight);
      progress.style.transform = `scaleX(${pct})`;
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  document.querySelectorAll('.article-content pre').forEach((pre) => {
    if (pre.querySelector('.code-copy')) return;
    const btn = document.createElement('button');
    btn.className = 'code-copy mono';
    btn.type = 'button';
    btn.textContent = 'copy';
    btn.addEventListener('click', () => {
      const code = pre.querySelector('code');
      const text = code ? code.innerText : pre.innerText;
      navigator.clipboard.writeText(text).then(() => {
        btn.textContent = 'copied!';
        setTimeout(() => (btn.textContent = 'copy'), 1800);
      });
    });
    pre.appendChild(btn);
  });

  // TOC active state
  const tocLinks = document.querySelectorAll('.toc a[href^="#"]');
  if (tocLinks.length) {
    const targets = [];
    tocLinks.forEach((a) => {
      const id = decodeURIComponent(a.getAttribute('href').slice(1));
      const t = document.getElementById(id);
      if (t) targets.push({ el: t, link: a });
    });
    if (targets.length) {
      const tocObs = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            tocLinks.forEach((l) => l.classList.remove('active'));
            const match = targets.find((t) => t.el === entry.target);
            if (match) match.link.classList.add('active');
          }
        });
      }, { rootMargin: '-20% 0px -70% 0px' });
      targets.forEach((t) => tocObs.observe(t.el));
    }
  }
})();
