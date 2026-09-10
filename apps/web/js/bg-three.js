// Subtle Three.js ambiance: a slow-drifting point field behind the content.
// Purely decorative, GPU-light (1 draw call), disabled when the user prefers
// reduced motion, and degrades silently if WebGL is unavailable.

import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";

const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
if (!reduced) {
  try { boot(); } catch { /* WebGL unavailable: static background is fine */ }
}

function boot() {
  const canvas = document.getElementById("bg-canvas");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 100);
  camera.position.z = 8;

  const COUNT = 900;
  const positions = new Float32Array(COUNT * 3);
  const speeds = new Float32Array(COUNT);
  for (let i = 0; i < COUNT; i++) {
    positions[i * 3] = (Math.random() - 0.5) * 24;
    positions[i * 3 + 1] = (Math.random() - 0.5) * 14;
    positions[i * 3 + 2] = (Math.random() - 0.5) * 8;
    speeds[i] = 0.0015 + Math.random() * 0.004;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const material = new THREE.PointsMaterial({
    color: 0x4ea1ff,
    size: 0.035,
    transparent: true,
    opacity: 0.55,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  const points = new THREE.Points(geometry, material);
  scene.add(points);

  const pointer = { x: 0, y: 0 };
  window.addEventListener("pointermove", (e) => {
    pointer.x = (e.clientX / window.innerWidth - 0.5) * 0.6;
    pointer.y = (e.clientY / window.innerHeight - 0.5) * 0.4;
  }, { passive: true });

  function resize() {
    renderer.setSize(window.innerWidth, window.innerHeight);
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
  }
  window.addEventListener("resize", resize);
  resize();

  let running = true;
  document.addEventListener("visibilitychange", () => {
    running = !document.hidden;
  });

  const clock = new THREE.Clock();
  function frame() {
    requestAnimationFrame(frame);
    if (!running) return;
    const dt = Math.min(clock.getDelta(), 0.05);
    const pos = geometry.attributes.position;
    for (let i = 0; i < COUNT; i++) {
      pos.array[i * 3 + 1] += speeds[i] * dt * 60 * 0.12;
      if (pos.array[i * 3 + 1] > 7) pos.array[i * 3 + 1] = -7;
    }
    pos.needsUpdate = true;
    points.rotation.y += dt * 0.02;
    camera.position.x += (pointer.x - camera.position.x) * 0.03;
    camera.position.y += (-pointer.y - camera.position.y) * 0.03;
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
  }
  frame();
}
