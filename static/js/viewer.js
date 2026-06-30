import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// ---- Demo metadata ----
const DEMOS = [
  { id: 'real_factory',     label: 'Real · Factory',     count: 4  },
  { id: 'real_indoor',      label: 'Real · Indoor',      count: 18 },
  { id: 'synthetic_indoor', label: 'Synthetic · Indoor', count: 39 },
  { id: 'weak_covis',       label: 'Weak Covisibility',  count: 2  },
  { id: 'curve_wall',       label: 'Curved Wall',        count: 1  },
  { id: 'cartoon',          label: 'Cartoon',            count: 1  },
  { id: 'ink_wash',         label: 'Ink Wash',           count: 1  },
  { id: 'animate',          label: 'Animation',          count: 1  },
];
const base = './static/demos';
const demoData = DEMOS.map(d => ({
  ...d,
  glb: `${base}/${d.id}/scene.glb`,
  images: Array.from({ length: d.count }, (_, i) => `${base}/${d.id}/img/v${i + 1}.jpg`),
}));

const canvasEl   = document.getElementById('viewerCanvas');
const tabsEl     = document.getElementById('demoTabs');
const inputsEl   = document.getElementById('demoInputs');
const loadingEl  = document.getElementById('viewerLoading');
if (canvasEl && tabsEl && inputsEl) {
  let renderer, scene, camera, controls, loader;
  let current = null, pointSize = 1.9, autoRotate = true, started = false;
  const cache = new Map();
  let activeId = demoData[0].id;

  const setLoading = (on) => { if (loadingEl) loadingEl.style.display = on ? 'flex' : 'none'; };

  function sizeOf() {
    const r = canvasEl.getBoundingClientRect();
    return { w: Math.max(1, r.width), h: Math.max(1, r.height) };
  }

  function init() {
    const { w, h } = sizeOf();
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w, h);
    canvasEl.appendChild(renderer.domElement);

    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(50, w / h, 0.01, 5000);
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.autoRotate = autoRotate;
    controls.autoRotateSpeed = 0.9;

    scene.add(new THREE.AmbientLight(0xffffff, 1.1));
    const dl = new THREE.DirectionalLight(0xffffff, 0.5);
    dl.position.set(1, 2, 1);
    scene.add(dl);

    window.addEventListener('resize', onResize);
    animate();
  }

  function onResize() {
    if (!renderer) return;
    const { w, h } = sizeOf();
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  function applyPointSize(group) {
    group.traverse((o) => {
      if (o.isPoints && o.material) {
        o.material.size = pointSize;
        o.material.sizeAttenuation = false;
        o.material.vertexColors = true;
        o.material.needsUpdate = true;
      }
    });
  }

  function frame(group) {
    const box = new THREE.Box3().setFromObject(group);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const radius = Math.max(size.x, size.y, size.z) * 0.5 || 1;
    controls.target.copy(center);
    const dir = new THREE.Vector3(1, 0.55, 1).normalize();
    camera.position.copy(center).add(dir.multiplyScalar(radius * 2.4));
    camera.near = radius / 100;
    camera.far = radius * 100;
    camera.updateProjectionMatrix();
    controls.update();
  }

  async function show(id) {
    activeId = id;
    [...tabsEl.children].forEach((b) => b.classList.toggle('active', b.dataset.id === id));
    renderInputs(id);
    if (!started) return;
    setLoading(true);
    const demo = demoData.find((d) => d.id === id);
    try {
      let group = cache.get(id);
      if (!group) {
        const gltf = await loader.loadAsync(demo.glb);
        group = gltf.scene;
        applyPointSize(group);
        cache.set(id, group);
      }
      if (current) scene.remove(current);
      current = group;
      scene.add(group);
      frame(group);
    } catch (e) {
      console.error('Failed to load demo', id, e);
    } finally {
      setLoading(false);
    }
  }

  function animate() {
    requestAnimationFrame(animate);
    if (controls) controls.update();
    if (renderer && scene && camera) renderer.render(scene, camera);
  }

  // ---- UI ----
  function renderTabs() {
    tabsEl.innerHTML = '';
    demoData.forEach((d) => {
      const b = document.createElement('button');
      b.className = 'demo-tab';
      b.dataset.id = d.id;
      b.textContent = d.label;
      b.addEventListener('click', () => show(d.id));
      tabsEl.appendChild(b);
    });
  }

  function renderInputs(id) {
    const demo = demoData.find((d) => d.id === id);
    inputsEl.innerHTML = '';
    demo.images.forEach((src) => {
      const im = document.createElement('img');
      im.src = src;
      im.loading = 'lazy';
      im.alt = 'Input panoramic view';
      inputsEl.appendChild(im);
    });
  }

  // ---- Controls wiring ----
  const reset = document.getElementById('vReset');
  const rotateBtn = document.getElementById('vRotate');
  const sizeSlider = document.getElementById('vPoint');
  if (reset) reset.addEventListener('click', () => { if (current) frame(current); });
  if (rotateBtn) rotateBtn.addEventListener('click', () => {
    autoRotate = !autoRotate;
    if (controls) controls.autoRotate = autoRotate;
    rotateBtn.textContent = autoRotate ? 'Pause rotation' : 'Auto-rotate';
  });
  if (sizeSlider) sizeSlider.addEventListener('input', (e) => {
    pointSize = parseFloat(e.target.value);
    if (current) applyPointSize(current);
  });

  // ---- Lazy start when scrolled into view ----
  function start() {
    if (started) return;
    started = true;
    loader = new GLTFLoader();
    init();
    show(activeId);
  }

  renderTabs();
  renderInputs(activeId);
  const io = new IntersectionObserver((entries) => {
    entries.forEach((en) => { if (en.isIntersecting) { start(); io.disconnect(); } });
  }, { rootMargin: '200px' });
  io.observe(canvasEl);
}
