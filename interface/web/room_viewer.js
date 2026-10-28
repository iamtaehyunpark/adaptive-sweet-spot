// 3D room viewer + speaker placement (Milestone 3.2b).
// Room frame: x = listener's right, y = toward the front, z = up (0 = face height), meters.
// three.js frame: x = room x, y = room z (up), z = -room y.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

const toThree = ([x, y, z]) => new THREE.Vector3(x, z, -y);
const toRoom = (v) => [v.x, -v.z, v.y];
const COLORS = { FL: 0x4c6ef5, FR: 0xf76707, FC: 0x2f9e44, SL: 0x7048e8, SR: 0xe8590c, LFE: 0x868e96 };
const FIXED = new Set(['FC']); // FC is the camera position

const statusEl = document.getElementById('status');
const status = (msg) => { statusEl.textContent = msg; };

// ---------------------------------------------------------------- data
const [room, xyzBuf, rgbBuf] = await Promise.all([
  fetch('/api/room').then((r) => r.json()),
  fetch('/api/cloud.xyz').then((r) => r.arrayBuffer()),
  fetch('/api/cloud.rgb').then((r) => r.arrayBuffer()),
]);
const xyz = new Float32Array(xyzBuf);
const rgb = new Uint8Array(rgbBuf);
const n = xyz.length / 3;
const floorZ = room.floor_z ?? -1.2;
const cameraXY = room.camera_xy;

// ---------------------------------------------------------------- scene
const view = document.getElementById('view');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
view.appendChild(renderer.domElement);
const labels = new CSS2DRenderer();
labels.setSize(window.innerWidth, window.innerHeight);
labels.domElement.style.position = 'fixed';
labels.domElement.style.inset = '0';
labels.domElement.style.pointerEvents = 'none';
view.appendChild(labels.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x15171a);
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.02, 100);
const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;
scene.add(new THREE.AmbientLight(0xffffff, 0.9));
const sun = new THREE.DirectionalLight(0xffffff, 0.8);
sun.position.set(2, 5, 3);
scene.add(sun);

// Point cloud, colored by the capture image.
const positions = new Float32Array(n * 3);
const colors = new Float32Array(n * 3);
for (let i = 0; i < n; i++) {
  positions[3 * i] = xyz[3 * i];
  positions[3 * i + 1] = xyz[3 * i + 2];
  positions[3 * i + 2] = -xyz[3 * i + 1];
  colors[3 * i] = rgb[3 * i] / 255;
  colors[3 * i + 1] = rgb[3 * i + 1] / 255;
  colors[3 * i + 2] = rgb[3 * i + 2] / 255;
}
const cloudGeom = new THREE.BufferGeometry();
cloudGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
cloudGeom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
const cloudMat = new THREE.PointsMaterial({ size: 0.01, vertexColors: true, sizeAttenuation: true });
const cloud = new THREE.Points(cloudGeom, cloudMat);
scene.add(cloud);

// Floor grid (1 m) and axes at the seat.
const grid = new THREE.GridHelper(12, 12, 0x5c636a, 0x343a40);
grid.position.y = floorZ;
scene.add(grid);

function label(text, cls = '') {
  const div = document.createElement('div');
  div.className = `label ${cls}`;
  div.textContent = text;
  return new CSS2DObject(div);
}

// Seat: a simple head + body down to the floor.
const seat = new THREE.Group();
const head = new THREE.Mesh(new THREE.SphereGeometry(0.09, 24, 16), new THREE.MeshStandardMaterial({ color: 0xd6336c }));
seat.add(head);
const bodyH = Math.max(-floorZ - 0.12, 0.1);
const body = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.12, bodyH, 16),
  new THREE.MeshStandardMaterial({ color: 0xd6336c, transparent: true, opacity: 0.45 }));
body.position.y = -0.12 - bodyH / 2;
seat.add(body);
const seatLabel = label('seat', 'seat');
seatLabel.position.set(0, 0.2, 0);
seat.add(seatLabel);
scene.add(seat);

// Camera marker (on FC) looking at the seat.
const camPos = toThree([cameraXY[0], cameraXY[1], 0]);
camPos.y = room.camera_height_m != null ? floorZ + room.camera_height_m : 0;
const camMarker = new THREE.Mesh(new THREE.ConeGeometry(0.06, 0.14, 4), new THREE.MeshStandardMaterial({ color: 0xffd43b }));
camMarker.position.copy(camPos);
camMarker.lookAt(new THREE.Vector3(0, 0, 0));
camMarker.rotateX(Math.PI / 2);
scene.add(camMarker);
const camLabel = label('camera');
camLabel.position.copy(camPos).add(new THREE.Vector3(0, 0.15, 0));
scene.add(camLabel);

// ---------------------------------------------------------------- speakers
let speakers = room.speakers.map((s) => ({ ...s, position: [...s.position] }));
const meshes = new Map();
const lines = new Map();
const lineMat = new THREE.LineBasicMaterial({ color: 0xadb5bd, transparent: true, opacity: 0.6 });

function speakerMesh(s) {
  const color = COLORS[s.name] ?? 0xffffff;
  const geom = s.lfe ? new THREE.BoxGeometry(0.3, 0.3, 0.3) : new THREE.BoxGeometry(0.14, 0.22, 0.14);
  const mesh = new THREE.Mesh(geom, new THREE.MeshStandardMaterial({ color, roughness: 0.5 }));
  mesh.userData.name = s.name;
  const lab = label(s.name);
  lab.position.set(0, s.lfe ? 0.25 : 0.2, 0);
  mesh.add(lab);
  return mesh;
}

function buildSpeakers() {
  for (const m of meshes.values()) { scene.remove(m); }
  for (const l of lines.values()) { scene.remove(l); }
  meshes.clear(); lines.clear();
  for (const s of speakers) {
    const mesh = speakerMesh(s);
    mesh.position.copy(toThree(s.position));
    scene.add(mesh);
    meshes.set(s.name, mesh);
    const geom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), mesh.position.clone()]);
    const line = new THREE.Line(geom, lineMat);
    line.visible = document.getElementById('showLines').checked;
    scene.add(line);
    lines.set(s.name, line);
  }
}

const transform = new TransformControls(camera, renderer.domElement);
transform.setSize(0.8);
transform.addEventListener('dragging-changed', (e) => { orbit.enabled = !e.value; });
transform.addEventListener('objectChange', () => {
  const s = speakers.find((sp) => sp.name === selected);
  if (!s) return;
  s.position = toRoom(transform.object.position).map((v) => Math.round(v * 1000) / 1000);
  s.placed = true;
  updateSpeaker(s);
});
scene.add(transform);

let selected = null;
function select(name, rerender = true) {
  selected = name;
  const mesh = meshes.get(name);
  if (mesh && !FIXED.has(name)) transform.attach(mesh); else transform.detach();
  if (rerender) {
    renderRows();
  } else {
    [...rowsEl.children].forEach((tr, i) => tr.classList.toggle('sel', speakers[i].name === name));
  }
}

function updateSpeaker(s) {
  const mesh = meshes.get(s.name);
  mesh.position.copy(toThree(s.position));
  const line = lines.get(s.name);
  line.geometry.setFromPoints([new THREE.Vector3(0, 0, 0), mesh.position.clone()]);
  updateReadout(s);
}

// ---------------------------------------------------------------- panel
const rowsEl = document.getElementById('rows');
const fmt = (v, d = 2) => (v >= 0 ? '+' : '') + v.toFixed(d);
function azDist(s) {
  const [x, y] = s.position;
  return `${fmt((Math.atan2(x, y) * 180) / Math.PI, 0)}&deg; / ${Math.hypot(x, y).toFixed(2)} m`;
}

function renderRows() {
  rowsEl.innerHTML = '';
  speakers.forEach((s, i) => {
    const tr = document.createElement('tr');
    if (s.name === selected) tr.className = 'sel';
    const color = '#' + (COLORS[s.name] ?? 0xffffff).toString(16).padStart(6, '0');
    const fixed = FIXED.has(s.name);
    tr.innerHTML = `<td class="name"><span class="swatch" style="background:${color}"></span>${i + 1} ${s.name}${fixed ? ' <span class="muted">(camera)</span>' : ''}</td>`
      + [0, 1, 2].map((k) => `<td><input type="number" step="0.01" data-k="${k}" value="${s.position[k].toFixed(2)}" ${fixed ? 'disabled' : ''}></td>`).join('')
      + `<td class="ro" id="ro-${s.name}">${azDist(s)}${s.placed ? '' : ' <span class="muted">default</span>'}</td>`;
    tr.addEventListener('click', (e) => { if (e.target.tagName !== 'INPUT') select(s.name); });
    tr.querySelectorAll('input').forEach((input) => {
      input.addEventListener('focus', () => { if (selected !== s.name) select(s.name, false); });
      input.addEventListener('change', () => {
        const v = parseFloat(input.value);
        if (Number.isFinite(v)) {
          s.position[+input.dataset.k] = v;
          s.placed = true;
          updateSpeaker(s);
        }
      });
    });
    rowsEl.appendChild(tr);
  });
}

function updateReadout(s) {
  const ro = document.getElementById(`ro-${s.name}`);
  if (ro) ro.innerHTML = azDist(s);
  const row = ro?.parentElement;
  row?.querySelectorAll('input').forEach((input) => {
    if (document.activeElement !== input) input.value = s.position[+input.dataset.k].toFixed(2);
  });
}

// ---------------------------------------------------------------- picking
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.02;
const pointer = new THREE.Vector2();
let downAt = null;
renderer.domElement.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4 || transform.dragging) return;
  pointer.set((e.clientX / window.innerWidth) * 2 - 1, -(e.clientY / window.innerHeight) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  if (e.shiftKey && selected && !FIXED.has(selected)) {
    const hit = raycaster.intersectObject(cloud)[0];
    if (hit) {
      const s = speakers.find((sp) => sp.name === selected);
      s.position = toRoom(hit.point).map((v) => Math.round(v * 1000) / 1000);
      s.placed = true;
      updateSpeaker(s);
      status(`${s.name} placed on the scan surface`);
    }
    return;
  }
  const hit = raycaster.intersectObjects([...meshes.values()], false)[0];
  if (hit) select(hit.object.userData.name);
});

window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  const i = parseInt(e.key, 10);
  if (i >= 1 && i <= speakers.length) select(speakers[i - 1].name);
  if (e.key === 'Escape') select(null);
});

// ---------------------------------------------------------------- views
function setView(name) {
  document.querySelectorAll('[data-view]').forEach((b) => b.classList.toggle('on', b.dataset.view === name));
  const front = toThree([0, Math.max(room.center_distance_m, 1), 0]);
  if (name === 'camera') {
    camera.position.copy(camPos);
    orbit.target.set(0, 0, 0);
  } else if (name === 'top') {
    camera.position.set(0, 7, 0.001);
    orbit.target.set(0, floorZ, front.z / 2);
    camera.position.z = front.z / 2 + 0.001;
  } else if (name === 'seat') {
    camera.position.set(0, 0.05, 0.35);
    orbit.target.copy(front);
  } else {
    camera.position.set(2.5, 2.5, 3.5);
    orbit.target.set(0, floorZ / 2, front.z / 2);
  }
  orbit.update();
}
document.querySelectorAll('[data-view]').forEach((b) => b.addEventListener('click', () => setView(b.dataset.view)));

// ---------------------------------------------------------------- controls
document.getElementById('psize').addEventListener('input', (e) => { cloudMat.size = e.target.value / 1000; });
document.getElementById('showLines').addEventListener('change', (e) => {
  for (const l of lines.values()) l.visible = e.target.checked;
});
document.getElementById('reset').addEventListener('click', async () => {
  speakers = (await fetch('/api/defaults').then((r) => r.json())).map((s) => ({ ...s, position: [...s.position] }));
  buildSpeakers();
  select(null);
  status('reset to defaults (not saved yet)');
});

async function post(path) {
  const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ speakers }) });
  return r.ok;
}
document.getElementById('save').addEventListener('click', async () => {
  status((await post('/api/save')) ? `saved ${new Date().toLocaleTimeString()}` : 'save failed');
});
document.getElementById('finish').addEventListener('click', async () => {
  if (await post('/api/finish')) {
    status('saved - you can close this tab');
    document.getElementById('finish').disabled = true;
  } else {
    status('save failed');
  }
});

const info = [`${n.toLocaleString()} points`, `scale x${room.scale.toFixed(3)} (${room.scale_source})`,
  `camera -> seat ${room.center_distance_m.toFixed(2)} m`];
if (room.camera_height_m != null) {
  info.push(`camera ${room.camera_height_m.toFixed(2)} m above floor, pitch ${fmt(room.camera_pitch_deg, 1)}&deg;`);
} else {
  info.push('no floor detected: level camera assumed');
}
document.getElementById('info').innerHTML = info.join(' &middot; ');

// ---------------------------------------------------------------- loop
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  labels.setSize(window.innerWidth, window.innerHeight);
});

buildSpeakers();
renderRows();
setView('camera');
status('Select a speaker, then drag its arrows or Shift+click the scan.');
renderer.setAnimationLoop(() => {
  orbit.update();
  renderer.render(scene, camera);
  labels.render(scene, camera);
});
