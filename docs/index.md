<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>NeLiF: Neural Lighting Function Generation for Indoor Rendering</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:        #0d0f14;
      --bg2:       #13161e;
      --bg3:       #1a1e28;
      --border:    rgba(255,255,255,0.07);
      --accent:    #c8a96e;
      --accent2:   #7eb8c9;
      --text:      #e8e6e0;
      --muted:     #8a8880;
      --font-serif: 'DM Serif Display', Georgia, serif;
      --font-sans:  'DM Sans', system-ui, sans-serif;
      --font-mono:  'JetBrains Mono', monospace;
      --radius:     10px;
      --max-w:      860px;
    }

    html { scroll-behavior: smooth; }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--font-sans);
      font-size: 16px;
      line-height: 1.75;
      -webkit-font-smoothing: antialiased;
    }

    /* ── Grain overlay ── */
    body::before {
      content: '';
      position: fixed; inset: 0;
      background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='1'/%3E%3C/svg%3E");
      opacity: 0.028;
      pointer-events: none;
      z-index: 999;
    }

    /* ── Nav ── */
    nav {
      position: sticky; top: 0; z-index: 100;
      background: rgba(13,15,20,0.85);
      backdrop-filter: blur(18px);
      -webkit-backdrop-filter: blur(18px);
      border-bottom: 1px solid var(--border);
      padding: 0 2rem;
    }
    .nav-inner {
      max-width: var(--max-w);
      margin: 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      height: 54px;
      gap: 1.5rem;
    }
    .nav-logo {
      font-family: var(--font-serif);
      font-size: 1.1rem;
      color: var(--accent);
      letter-spacing: 0.02em;
      white-space: nowrap;
    }
    .nav-links {
      display: flex; gap: 0.25rem;
      list-style: none;
      flex-wrap: wrap;
    }
    .nav-links a {
      color: var(--muted);
      text-decoration: none;
      font-size: 0.82rem;
      font-weight: 500;
      padding: 0.3rem 0.7rem;
      border-radius: 6px;
      transition: color 0.2s, background 0.2s;
      letter-spacing: 0.03em;
    }
    .nav-links a:hover { color: var(--text); background: var(--bg3); }

    /* ── Sections ── */
    section { padding: 5rem 2rem; }
    section + section { padding-top: 0; }
    .section-inner { max-width: var(--max-w); margin: 0 auto; }

    /* ── Hero ── */
    #hero {
      padding: 5rem 2rem 3.5rem;
      background: radial-gradient(ellipse 80% 60% at 50% -10%, rgba(200,169,110,0.10) 0%, transparent 70%);
    }
    .venue-tag {
      display: inline-block;
      font-family: var(--font-mono);
      font-size: 0.75rem;
      color: var(--accent);
      background: rgba(200,169,110,0.10);
      border: 1px solid rgba(200,169,110,0.25);
      padding: 0.3rem 0.8rem;
      border-radius: 4px;
      letter-spacing: 0.08em;
      margin-bottom: 1.5rem;
      animation: fadeUp 0.6s ease both;
    }
    h1.paper-title {
      font-family: var(--font-serif);
      font-size: clamp(2rem, 5vw, 2.9rem);
      line-height: 1.2;
      color: #fff;
      margin-bottom: 0.5rem;
      animation: fadeUp 0.6s 0.1s ease both;
    }
    h1.paper-title em {
      font-style: italic;
      color: var(--accent);
    }

    /* ── Authors ── */
    .authors-block {
      margin: 1.8rem 0 1.4rem;
      animation: fadeUp 0.6s 0.2s ease both;
    }
    .authors-list {
      display: flex; flex-wrap: wrap; gap: 0.3rem 0.15rem;
      font-size: 0.92rem;
      color: var(--text);
      line-height: 2;
    }
    .author { white-space: nowrap; }
    .author sup { color: var(--accent2); font-size: 0.7em; margin-left: 1px; }
    .author .sym { color: var(--accent); font-size: 0.75em; margin-right: 1px; }
    .author-sep { color: var(--border); margin: 0 0.1rem; }

    .author-note {
      font-size: 0.78rem;
      color: var(--muted);
      margin-top: 0.5rem;
      font-style: italic;
    }

    .affiliations {
      margin-top: 1rem;
      font-size: 0.83rem;
      color: var(--muted);
      display: flex; flex-direction: column; gap: 0.2rem;
      animation: fadeUp 0.6s 0.25s ease both;
    }
    .affiliations span { display: flex; align-items: baseline; gap: 0.35rem; }
    .affiliations sup { color: var(--accent2); font-size: 0.72em; }

    /* ── Link Buttons ── */
    .btn-row {
      display: flex; flex-wrap: wrap; gap: 0.6rem;
      margin-top: 2rem;
      animation: fadeUp 0.6s 0.3s ease both;
    }
    .btn {
      display: inline-flex; align-items: center; gap: 0.45rem;
      padding: 0.55rem 1.15rem;
      border-radius: var(--radius);
      font-size: 0.84rem;
      font-weight: 600;
      text-decoration: none;
      letter-spacing: 0.03em;
      transition: transform 0.18s, box-shadow 0.18s, background 0.18s;
      cursor: pointer;
    }
    .btn:hover { transform: translateY(-2px); }
    .btn-primary {
      background: var(--accent);
      color: #0d0f14;
      box-shadow: 0 4px 20px rgba(200,169,110,0.25);
    }
    .btn-primary:hover { background: #d9bc85; box-shadow: 0 6px 28px rgba(200,169,110,0.35); }
    .btn-secondary {
      background: var(--bg3);
      color: var(--text);
      border: 1px solid var(--border);
    }
    .btn-secondary:hover { background: #21263a; border-color: rgba(255,255,255,0.14); }
    .btn svg { width: 15px; height: 15px; flex-shrink: 0; }

    /* ── Section Labels ── */
    .sec-label {
      font-family: var(--font-mono);
      font-size: 0.7rem;
      letter-spacing: 0.14em;
      color: var(--accent);
      text-transform: uppercase;
      margin-bottom: 0.6rem;
    }
    h2.sec-title {
      font-family: var(--font-serif);
      font-size: clamp(1.5rem, 3vw, 1.9rem);
      color: #fff;
      margin-bottom: 1.5rem;
      line-height: 1.3;
    }

    /* ── Divider ── */
    .divider {
      max-width: var(--max-w);
      margin: 0 auto;
      border: none;
      border-top: 1px solid var(--border);
    }

    /* ── Teaser ── */
    .teaser-wrap {
      border-radius: 12px;
      overflow: hidden;
      border: 1px solid var(--border);
      background: var(--bg2);
      position: relative;
    }
    .teaser-wrap img {
      width: 100%; display: block;
      transition: transform 0.5s ease;
    }
    .teaser-wrap:hover img { transform: scale(1.01); }
    .teaser-caption {
      padding: 1rem 1.25rem;
      font-size: 0.85rem;
      color: var(--muted);
      border-top: 1px solid var(--border);
      line-height: 1.6;
    }

    /* ── Abstract ── */
    .abstract-text {
      font-size: 0.97rem;
      color: var(--text);
      line-height: 1.85;
      background: var(--bg2);
      border: 1px solid var(--border);
      border-left: 3px solid var(--accent);
      padding: 1.6rem 1.8rem;
      border-radius: var(--radius);
    }

    /* ── Video ── */
    .video-wrap {
      position: relative;
      width: 100%; padding-bottom: 56.25%;
      border-radius: 12px; overflow: hidden;
      border: 1px solid var(--border);
      background: var(--bg2);
    }
    .video-wrap iframe {
      position: absolute; inset: 0;
      width: 100%; height: 100%; border: none;
    }
    .video-placeholder {
      position: absolute; inset: 0;
      display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      gap: 0.75rem;
      color: var(--muted);
      font-size: 0.88rem;
    }
    .play-icon {
      width: 52px; height: 52px;
      border: 2px solid var(--border);
      border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      color: var(--muted);
    }

    /* ── Method ── */
    .pipeline-wrap {
      border-radius: 12px; overflow: hidden;
      border: 1px solid var(--border);
      background: var(--bg2);
      margin-bottom: 1.6rem;
    }
    .pipeline-wrap img { width: 100%; display: block; }
    .key-points {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 1rem; margin-top: 1.2rem;
    }
    .key-point {
      background: var(--bg2);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.2rem 1.3rem;
      transition: border-color 0.2s, transform 0.2s;
    }
    .key-point:hover { border-color: rgba(200,169,110,0.3); transform: translateY(-2px); }
    .kp-num {
      font-family: var(--font-mono);
      font-size: 0.7rem;
      color: var(--accent);
      letter-spacing: 0.1em;
      margin-bottom: 0.5rem;
    }
    .key-point p { font-size: 0.88rem; color: var(--muted); line-height: 1.65; }
    .key-point strong { color: var(--text); font-weight: 600; }

    /* ── Results ── */
    .result-fig {
      border-radius: 12px; overflow: hidden;
      border: 1px solid var(--border);
      background: var(--bg2);
      margin-bottom: 0.75rem;
    }
    .result-fig img { width: 100%; display: block; }
    .fig-caption {
      font-size: 0.83rem; color: var(--muted);
      padding: 0.8rem 1.1rem;
      border-top: 1px solid var(--border);
      line-height: 1.6;
    }

    /* ── Comparison Table ── */
    .table-wrap { overflow-x: auto; margin-top: 1.5rem; }
    table {
      width: 100%; border-collapse: collapse;
      font-size: 0.85rem;
    }
    thead th {
      background: var(--bg3);
      color: var(--accent);
      font-family: var(--font-mono);
      font-size: 0.73rem;
      letter-spacing: 0.07em;
      padding: 0.75rem 1rem;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }
    tbody td {
      padding: 0.7rem 1rem;
      border-bottom: 1px solid var(--border);
      color: var(--text);
    }
    tbody tr:last-child td { border-bottom: none; }
    tbody tr:hover td { background: var(--bg3); }
    .best { color: var(--accent); font-weight: 700; }
    .ours-row td { background: rgba(200,169,110,0.05); }

    /* ── Usage ── */
    .code-block {
      background: var(--bg2);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow-x: auto;
      position: relative;
    }
    .code-block pre {
      font-family: var(--font-mono);
      font-size: 0.82rem;
      color: #b8d4c0;
      padding: 1.4rem 1.6rem;
      line-height: 1.7;
      white-space: pre;
    }
    .code-block .line-comment { color: #556a5b; }
    .code-block .line-cmd { color: #7eb8c9; }


    #usage p { margin: 1rem 0; }
    #usage a { color: var(--accent2); overflow-wrap: anywhere; }
    #usage p code, #usage td code {
      font-family: var(--font-mono);
      font-size: 0.85em;
      overflow-wrap: anywhere;
    }
    .usage-step { margin-top: 2.5rem; }
    .usage-step h3 {
      font-family: var(--font-serif);
      font-size: 1.2rem;
      font-weight: 400;
      color: #fff;
      margin-bottom: 1rem;
    }
    #usage .code-block + .code-block { margin-top: 1rem; }

    /* ── Citation ── */
    .citation-block {
      position: relative;
      background: var(--bg2);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.4rem 1.6rem;
      font-family: var(--font-mono);
      font-size: 0.8rem;
      color: #9cb8a8;
      line-height: 1.75;
      overflow-x: auto;
      white-space: pre;
    }
    .copy-btn {
      position: absolute; top: 0.8rem; right: 0.8rem;
      background: var(--bg3);
      border: 1px solid var(--border);
      color: var(--muted);
      font-family: var(--font-mono);
      font-size: 0.72rem;
      padding: 0.3rem 0.7rem;
      border-radius: 6px;
      cursor: pointer;
      transition: color 0.2s, background 0.2s;
      letter-spacing: 0.05em;
    }
    .copy-btn:hover { color: var(--text); background: #21263a; }

    /* ── Footer ── */
    footer {
      padding: 2.5rem 2rem;
      border-top: 1px solid var(--border);
      text-align: center;
      color: var(--muted);
      font-size: 0.8rem;
      line-height: 1.8;
    }
    footer a { color: var(--accent2); text-decoration: none; }
    footer a:hover { text-decoration: underline; }

    /* ── Animations ── */
    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(18px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    .reveal {
      opacity: 0; transform: translateY(20px);
      transition: opacity 0.55s ease, transform 0.55s ease;
    }
    .reveal.visible { opacity: 1; transform: none; }

    /* ── Responsive ── */
    @media (max-width: 600px) {
      .nav-links { display: none; }
      h1.paper-title { font-size: 1.65rem; }
      section { padding: 3.5rem 1.2rem; }
      .abstract-text { padding: 1.2rem 1.3rem; }
    }
  </style>
</head>
<body>

<!-- ════════════════════════ NAV ════════════════════════ -->
<nav>
  <div class="nav-inner">
    <span class="nav-logo">NeLiF</span>
    <ul class="nav-links">
      <li><a href="#teaser">Teaser</a></li>
      <li><a href="#abstract">Abstract</a></li>
      <li><a href="#video">Video</a></li>
      <li><a href="#method">Method</a></li>
      <li><a href="#results">Results</a></li>
      <li><a href="#usage">Code</a></li>
      <li><a href="#citation">Citation</a></li>
    </ul>
  </div>
</nav>

<!-- ════════════════════════ HERO ════════════════════════ -->
<section id="hero">
  <div class="section-inner">

    <div class="venue-tag">SIGGRAPH Asia 2025 &nbsp;&bull;&nbsp; to appear</div>

    <h1 class="paper-title">
      NeLiF: Neural Lighting Function<br>Generation for Indoor Rendering
    </h1>

    <!-- Authors -->
    <div class="authors-block">
      <div class="authors-list">
        <span class="author"><span class="sym">*</span>Hongtao Sheng<sup>1</sup></span>
        <span class="author-sep">,</span>
        <span class="author"><span class="sym">*</span>Yuchi Huo<sup>1</sup></span>
        <span class="author-sep">,</span>
        <span class="author">Chuankun Zheng<sup>1</sup></span>
        <span class="author-sep">,</span>
        <span class="author">Yifan Peng<sup>2</sup></span>
        <span class="author-sep">,</span>
        <span class="author">Guangzhi Han<sup>1</sup></span>
        <span class="author-sep">,</span>
        <span class="author">Shi Li<sup>1</sup></span>
        <span class="author-sep">,</span>
        <span class="author">Bin Zang<sup>1</sup></span>
        <span class="author-sep">,</span>
        <span class="author">Rui Wang<sup>1</sup></span>
        <span class="author-sep">,</span>
        <span class="author">Hao Zhu<sup>3</sup></span>
        <span class="author-sep">,</span>
        <span class="author">Rui Tang<sup>3</sup></span>
        <span class="author-sep">,</span>
        <span class="author">Yiming Wu<sup>3</sup></span>
        <span class="author"><span class="sym" style="color:var(--accent2)">†</span>Hujun Bao<sup>1</sup></span>
        <span class="author-sep">,</span>
      </div>
      <div class="author-note">* Joint first authors &nbsp;&nbsp; † Corresponding author</div>
    </div>

    <!-- Affiliations -->
    <div class="affiliations">
      <span><sup>1</sup> State Key Lab of CAD&amp;CG, Zhejiang University, China</span>
      <span><sup>2</sup> The University of Hong Kong, China</span>
      <span><sup>3</sup> Manycore Tech Inc., China</span>
    </div>

    <!-- Buttons -->
    <div class="btn-row">
      <a href="https://dl.acm.org/doi/10.1145/3757377.3763958" target="_blank" class="btn btn-primary">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
        Paper
      </a>
      <a href="https://github.com/GensokyoLover/NeLiF-Neural-Lighting-Function-Generation-for-Indoor-Rendering" class="btn btn-secondary">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"/></svg>
        Code
      </a>
      <a href="https://huggingface.co/datasets/GensokyoLOvEr/NeLiF-Neural-Lighting-Function-Generation-for-Indoor-Rendering/tree/main" class="btn btn-secondary">Data &amp; Model</a>
    </div>

  </div>
</section>

<hr class="divider" />

<!-- ════════════════════════ TEASER ════════════════════════ -->
<section id="teaser">
  <div class="section-inner reveal">
    <div class="sec-label">Overview</div>
    <h2 class="sec-title">Teaser</h2>
    <div class="teaser-wrap">
      <img src="assets/teaser.png" alt="NeLiF Teaser — Neural Lighting Function for Indoor Rendering" />
      <div class="teaser-caption">
        <strong>Figure 1.</strong> Replace this caption with 1–2 sentences describing what is shown in your teaser image — e.g., a comparison between ground truth, baseline methods, and NeLiF results under complex indoor luminaire configurations.
      </div>
    </div>
  </div>
</section>

<hr class="divider" />

<!-- ════════════════════════ ABSTRACT ════════════════════════ -->
<section id="abstract">
  <div class="section-inner reveal">
    <div class="sec-label">Abstract</div>
    <h2 class="sec-title">Abstract</h2>
    <div class="abstract-text">
      Recent advances in neural rendering have mainly focused on modeling radiance fields with neural representations, often overlooking the underlying mechanisms for producing various lighting effects, and consequently leading to the limited adaptability to dynamic scenes. Lighting effects, such as highlights, shadows, and indirect illuminations, are typically computed using physically-based rendering methods like path tracing, which can be computationally intensive for complex indoor luminaires. Although several recent studies have aimed to model global illumination effects with neural representations, they commonly suffer from long training times or poor generalizability to new scenes. Addressing these challenges, this work presents a novel neural lighting function generation model capable of synthesizing diverse lighting effects in real time for unseen dynamic scenes and complex indoor luminaires, achieving results comparable to state-of-the-art rendering pipelines. Our model operates in two stages. First, multi-view observation images of the luminaire are captured to encode a compact, scene-independent 3D neural lighting field. Subsequently, light information is sampled from this neural lighting field and integrated with G-buffers and shadow clues to produce the shading results. In parallel, we employ a state-of-the-art generative model together with our training-free Inverse HDR Splatting module to generate HDR 3D Gaussians representing the luminaire. This strategy capitalizes on the powerful generalization capabilities of advanced generative models, enabling efficient and accurate appearance reconstruction for a diverse range of complex luminaires. In our experiments, the model trained on a dataset of 10,000 modern indoor scenes and thousands of illuminations demonstrates strong generalizability, high efficiency, and visually convincing results across a wide range of test scenes, highlighting its potential as a practical and flexible solution for high-fidelity, real-time neural indoor rendering.
    </div>
  </div>
</section>

<hr class="divider" />

<!-- ════════════════════════ VIDEO ════════════════════════ -->
<section id="video">
  <div class="section-inner reveal">
    <div class="sec-label">Demo</div>
    <h2 class="sec-title">Video</h2>
    <div class="video-wrap">
      <!--
        Replace the placeholder below with your actual embed.
        YouTube: <iframe src="https://www.youtube.com/embed/YOUR_VIDEO_ID" allowfullscreen></iframe>
        Bilibili: <iframe src="https://player.bilibili.com/player.html?bvid=YOUR_BVID" allowfullscreen></iframe>
      -->
      <div class="video-placeholder">
        <div class="play-icon">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        </div>
        <span>Replace with your YouTube / Bilibili embed URL</span>
        <span style="font-size:0.76rem; font-family: var(--font-mono); color: rgba(138,136,128,0.5)">e.g., https://www.youtube.com/embed/YOUR_VIDEO_ID</span>
      </div>
    </div>
  </div>
</section>

<hr class="divider" />

<!-- ════════════════════════ METHOD ════════════════════════ -->
<section id="method">
  <div class="section-inner reveal">
    <div class="sec-label">Approach</div>
    <h2 class="sec-title">Method Overview</h2>

    <div class="pipeline-wrap">
      <img src="assets/pipeline.png" alt="NeLiF Pipeline Figure" />
    </div>

    <div class="key-points">
      <div class="key-point">
        <div class="kp-num">01 / KEY IDEA</div>
        <p><strong>Replace with your key idea 1.</strong> A brief description of the first main component or contribution of your method.</p>
      </div>
      <div class="key-point">
        <div class="kp-num">02 / KEY IDEA</div>
        <p><strong>Replace with your key idea 2.</strong> A brief description of the second main component or contribution of your method.</p>
      </div>
      <div class="key-point">
        <div class="kp-num">03 / KEY IDEA</div>
        <p><strong>Replace with your key idea 3.</strong> A brief description of the third main component or contribution of your method.</p>
      </div>
    </div>
  </div>
</section>

<hr class="divider" />

<!-- ════════════════════════ RESULTS ════════════════════════ -->
<section id="results">
  <div class="section-inner">

    <div class="sec-label">Evaluation</div>
    <h2 class="sec-title">Results</h2>

    <!-- Qualitative -->
    <div class="reveal">
      <h3 style="font-family:var(--font-serif); font-size:1.15rem; color:#fff; margin-bottom:1rem; font-weight:400;">
        Qualitative Results
      </h3>
      <div class="result-fig">
        <img src="assets/results1.png" alt="NeLiF Qualitative Results" />
        <div class="fig-caption">
          <strong>Figure 2.</strong> Replace with a description of your qualitative results, e.g., comparison of rendered images across different indoor scenes.
        </div>
      </div>
    </div>

    <!-- Comparisons -->
    <div class="reveal" style="margin-top:2.5rem;">
      <h3 style="font-family:var(--font-serif); font-size:1.15rem; color:#fff; margin-bottom:1rem; font-weight:400;">
        Comparisons &amp; Ablations
      </h3>
      <div class="result-fig">
        <img src="assets/results2.png" alt="NeLiF Ablation Results" />
        <div class="fig-caption">
          <strong>Figure 3.</strong> Replace with a description of your ablation or comparison figures.
        </div>
      </div>

      <!-- Quantitative Table -->
      <div class="table-wrap" style="margin-top:1.5rem;">
        <table>
          <thead>
            <tr>
              <th>Method</th>
              <th>PSNR (dB) &uarr;</th>
              <th>SSIM &uarr;</th>
              <th>LPIPS &darr;</th>
              <th>Time (ms) &darr;</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Baseline A</td>
              <td>28.4</td><td>0.871</td><td>0.142</td><td>38.2</td>
            </tr>
            <tr>
              <td>Baseline B</td>
              <td>30.1</td><td>0.889</td><td>0.118</td><td>52.7</td>
            </tr>
            <tr>
              <td>Baseline C</td>
              <td>31.6</td><td>0.901</td><td>0.097</td><td>61.3</td>
            </tr>
            <tr class="ours-row">
              <td><strong>NeLiF (Ours)</strong></td>
              <td class="best">34.2</td>
              <td class="best">0.932</td>
              <td class="best">0.071</td>
              <td class="best">29.4</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p style="font-size:0.78rem; color:var(--muted); margin-top:0.6rem;">
        Table 1. Replace these placeholder numbers with your actual quantitative results.
        Bold values indicate best performance.
      </p>
    </div>

  </div>
</section>

<hr class="divider" />

<!-- ════════════════════════ USAGE ════════════════════════ -->
<section id="usage">
  <div class="section-inner">
    <div class="sec-label">Code &amp; Data</div>
    <h2 class="sec-title">Run NeLiF</h2>

    <div class="usage-step reveal">
      <h3>1. Clone and install</h3>
      <p>Run these commands from a terminal with Conda available. The environment uses Python 3.10, PyTorch 2.6.0, and CUDA 12.6.</p>
      <div class="code-block"><pre><code>git clone https://github.com/GensokyoLover/NeLiF-Neural-Lighting-Function-Generation-for-Indoor-Rendering.git
cd NeLiF-Neural-Lighting-Function-Generation-for-Indoor-Rendering

conda env create -f environment.yml
conda activate nelif</code></pre></div>

    </div>

    <div class="usage-step reveal">
      <h3>2. Download the data and pretrained model</h3>
      <p>Download the files below from <a href="https://huggingface.co/datasets/GensokyoLOvEr/NeLiF-Neural-Lighting-Function-Generation-for-Indoor-Rendering/tree/main">Hugging Face</a>.</p>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Download</th><th>Size</th><th>Final location</th></tr></thead>
          <tbody>
            <tr><td><a href="https://huggingface.co/datasets/GensokyoLOvEr/NeLiF-Neural-Lighting-Function-Generation-for-Indoor-Rendering/resolve/main/scene.tar.gz?download=true">scene.tar.gz</a></td><td>6.24 GB</td><td><code>datasets/scene/*.pkl.zst</code></td></tr>
            <tr><td><a href="https://huggingface.co/datasets/GensokyoLOvEr/NeLiF-Neural-Lighting-Function-Generation-for-Indoor-Rendering/resolve/main/light.zip?download=true">light.zip</a></td><td>76.6 MB</td><td><code>datasets/Light/*.pkl.zst</code></td></tr>
            <tr><td><a href="https://huggingface.co/datasets/GensokyoLOvEr/NeLiF-Neural-Lighting-Function-Generation-for-Indoor-Rendering/resolve/main/model.pt?download=true">model.pt</a></td><td>327.8 MB</td><td><code>ckpts/model.pt</code></td></tr>
          </tbody>
        </table>
      </div>
      <p><code>datasets/OutDir.exr</code>, <code>datasets/indirect_dir.exr</code>, and <code>datasets/bias_info.json</code> are included in Git.</p>
      <p>Download using the links above or the <a href="https://huggingface.co/docs/huggingface_hub/guides/cli">Hugging Face CLI</a>, then extract the archives:</p>
      <div class="code-block"><pre><code>python -m pip install -U huggingface_hub
hf download GensokyoLOvEr/NeLiF-Neural-Lighting-Function-Generation-for-Indoor-Rendering scene.tar.gz light.zip --repo-type dataset --local-dir downloads
hf download GensokyoLOvEr/NeLiF-Neural-Lighting-Function-Generation-for-Indoor-Rendering model.pt --repo-type dataset --local-dir ckpts

python -m tarfile -e downloads/scene.tar.gz downloads/scene_unpacked
python -m zipfile -e downloads/light.zip downloads/light_unpacked</code></pre></div>
      <p>Place the extracted <code>.pkl.zst</code> files directly in <code>datasets/scene/</code> and <code>datasets/Light/</code>, preserving their file names and the directory capitalization shown below.</p>

    </div>

    <div class="usage-step reveal">
      <h3>3. Check the directory structure</h3>
      <div class="code-block"><pre><code>NeLiF-Neural-Lighting-Function-Generation-for-Indoor-Rendering/
├── README.md
├── configs/
│   └── nelif/
│       └── nelif.json                 # Model/evaluation configuration (Git)
├── ckpts/
│   └── model.pt                       # Download from Hugging Face
├── datasets/
│   ├── OutDir.exr                     # Included in Git
│   ├── indirect_dir.exr               # Included in Git
│   ├── bias_info.json                 # Included in Git
│   ├── scene/
│   │   └── &lt;scene_sample&gt;.pkl.zst      # Extract scene.tar.gz here
│   └── Light/
│       └── &lt;lightID&gt;.pkl.zst           # Extract light.zip here
├── src/
│   ├── nelif_run.py                   # Inference entry point
│   ├── dataset.py
│   ├── common/
│   ├── networks/
│   └── utils/
├── docs/
│   └── index.md                       # Paper project page
└── outputs/                           # Created when running inference</code></pre></div>

    </div>

    <div class="usage-step reveal">
      <h3>4. Run inference</h3>
      <p>From the repository root, run:</p>
      <div class="code-block"><pre><code>python src/nelif_run.py --config configs/nelif/nelif.json --ckpt_path ckpts/model.pt --label base --job_name base --light_angular_resolution 8 --light_direction_resolution 128 --diffuse --specular --shadow --indirect --device cuda --output_dir outputs/nelif_test</code></pre></div>
      <p>This runs all four rendering branches for every scene, saves their results, and reports PSNR.</p>

    </div>

    <div class="usage-step reveal">
      <h3>5. View the saved results</h3>
      <p>Each scene has its own output directory:</p>
      <div class="code-block"><pre><code>outputs/nelif_test/test/data/00000_&lt;scene_sample&gt;/
  pred_shading_&lt;N&gt;.exr       # Combined prediction
  shading_&lt;N&gt;.exr            # Ground truth
  pred_shadow_&lt;N&gt;.exr
  pred_indirect_shading_&lt;N&gt;.exr
  ...                       # Direct components, ground truth, and mask</code></pre></div>
      <p><code>&lt;N&gt;</code> is the sample counter, starting at 1. Results are saved as HDR EXR images. Use <code>--no_save</code> to compute metrics without saving images.</p>
      <p>See the <a href="https://github.com/GensokyoLover/NeLiF-Neural-Lighting-Function-Generation-for-Indoor-Rendering#readme">README</a> for option details, or run <code>python src/nelif_run.py --help</code>.</p>
    </div>
  </div>
</section>

<hr class="divider" />

<!-- ════════════════════════ CITATION ════════════════════════ -->
<section id="citation">
  <div class="section-inner reveal">
    <div class="sec-label">Reference</div>
    <h2 class="sec-title">Citation</h2>
    <div style="position:relative;">
      <div class="citation-block" id="bibtex-text">@article{sheng2025nelif,
  title     = {NeLiF: Neural Lighting Function Generation for Indoor Rendering},
  author    = {Sheng, Hongtao and Huo, Yuchi and Zheng, Chuankun and
               Han, Guangzhi and Li, Shi and Zang, Bin and Wang, Rui and
               Bao, Hujun and Peng, Yifan and Zhu, Hao and Tang, Rui and Wu, Yiming},
  journal   = {ACM Transactions on Graphics (SIGGRAPH Asia 2025)},
  year      = {2025},
}</div>
      <button class="copy-btn" onclick="copyBibTeX(this)">copy</button>
    </div>
  </div>
</section>

<!-- ════════════════════════ FOOTER ════════════════════════ -->
<footer>
  <p>NeLiF &mdash; SIGGRAPH Asia 2025</p>
  <p style="margin-top:0.3rem;">
    <a href="https://www.cad.zju.edu.cn/" target="_blank">State Key Lab of CAD&amp;CG, Zhejiang University</a>
    &nbsp;&bull;&nbsp;
    <a href="https://www.hku.hk/" target="_blank">The University of Hong Kong</a>
    &nbsp;&bull;&nbsp;
    <a href="https://www.manycoretech.com/" target="_blank">Manycore Tech Inc.</a>
  </p>
  <p style="margin-top:0.5rem; color: rgba(138,136,128,0.4); font-size:0.73rem;">
    Page template inspired by academic project pages.
  </p>
</footer>

<script>
  // ── Scroll reveal ──
  const revealEls = document.querySelectorAll('.reveal');
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((e, i) => {
      if (e.isIntersecting) {
        // stagger sibling reveals
        const siblings = [...e.target.parentElement.querySelectorAll('.reveal')];
        const idx = siblings.indexOf(e.target);
        e.target.style.transitionDelay = (idx * 0.08) + 's';
        e.target.classList.add('visible');
        observer.unobserve(e.target);
      }
    });
  }, { threshold: 0.1 });
  revealEls.forEach(el => observer.observe(el));

  // ── Copy BibTeX ──
  function copyBibTeX(btn) {
    const text = document.getElementById('bibtex-text').innerText;
    navigator.clipboard.writeText(text).then(() => {
      btn.textContent = 'copied!';
      btn.style.color = 'var(--accent)';
      setTimeout(() => {
        btn.textContent = 'copy';
        btn.style.color = '';
      }, 2000);
    });
  }
</script>

</body>
</html>
