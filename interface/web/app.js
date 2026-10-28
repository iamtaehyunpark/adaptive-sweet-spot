// Sweet Spot Live: 3D room scan + live listener + top view + speaker placement + setup.
// Room frame: x = listener's right, y = toward the front, z = up (0 = face height), meters.
// three.js frame: x = room x, y = room z (up), z = -room y.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

const toThree = ([x, y, z]) => new THREE.Vector3(x, z, -y);
const toRoom = (v) => [v.x, -v.z, v.y];
const COLORS = { FL: 0x4c6ef5, FR: 0xf76707, FC: 0x2f9e44, SL: 0x7048e8, SR: 0xe8590c, LFE: 0x868e96 };
const hex = (c) => '#' + c.toString(16).padStart(6, '0');
const FIXED = new Set(['FC']); // FC is the camera position
const TRAIL_LEN = 90;
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 2) => (v >= 0 ? '+' : '') + v.toFixed(d);

// ---------------------------------------------------------------- tabs
document.querySelectorAll('[data-tab]').forEach((b) => b.addEventListener('click', () => {
  document.querySelectorAll('[data-tab]').forEach((x) => x.classList.toggle('on', x === b));
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('on', t.id === `tab-${b.dataset.tab}`));
}));

// ---------------------------------------------------------------- renderer / scene
const view = $('view');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
view.appendChild(renderer.domElement);
const labels = new CSS2DRenderer();
labels.setSize(window.innerWidth, window.innerHeight);
Object.assign(labels.domElement.style, { position: 'fixed', inset: '0', pointerEvents: 'none' });
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

function label(text, cls = '') {
  const div = document.createElement('div');
  div.className = `label ${cls}`;
  div.textContent = text;
  return new CSS2DObject(div);
}

function disposeGroup(group) {
  group.traverse((o) => {
    o.geometry?.dispose();
    if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => m.dispose());
    if (o.isCSS2DObject) o.element.remove();
  });
  scene.remove(group);
}

// ---------------------------------------------------------------- room (scan, seat, camera)
let room = null;
let roomGroup = new THREE.Group();
let cloud = null;
let floorZ = -1.2;
let camPos = new THREE.Vector3(0, 0, -2);
const cloudMat = new THREE.PointsMaterial({ size: 0.01, vertexColors: true, sizeAttenuation: true });
let mapImg = null;

async function loadRoom() {
  room = await fetch('/api/room').then((r) => r.json());
  disposeGroup(roomGroup);
  roomGroup = new THREE.Group();
  scene.add(roomGroup);
  cloud = null;
  floorZ = room.floor_z ?? -1.2;

  if (room.ready) {
    const [xyzBuf, rgbBuf] = await Promise.all([
      fetch('/api/cloud.xyz').then((r) => r.arrayBuffer()),
      fetch('/api/cloud.rgb').then((r) => r.arrayBuffer()),
    ]);
    const xyz = new Float32Array(xyzBuf);
    const rgb = new Uint8Array(rgbBuf);
    const n = xyz.length / 3;
    const pos = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      pos[3 * i] = xyz[3 * i]; pos[3 * i + 1] = xyz[3 * i + 2]; pos[3 * i + 2] = -xyz[3 * i + 1];
      col[3 * i] = rgb[3 * i] / 255; col[3 * i + 1] = rgb[3 * i + 1] / 255; col[3 * i + 2] = rgb[3 * i + 2] / 255;
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(col, 3));
    cloud = new THREE.Points(geom, cloudMat);
    roomGroup.add(cloud);
    mapImg = new Image();
    mapImg.src = `/api/map.png?v=${Date.now()}`;
  } else {
    mapImg = null;
  }
  $('empty').style.display = room.ready ? 'none' : 'flex';

  const grid = new THREE.GridHelper(12, 12, 0x5c636a, 0x343a40);
  grid.position.y = floorZ;
  roomGroup.add(grid);

  // Seat: where the listener sat during setup (origin).
  const seat = new THREE.Group();
  const bodyH = Math.max(-floorZ - 0.12, 0.1);
  const body = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.12, bodyH, 16),
    new THREE.MeshStandardMaterial({ color: 0xd6336c, transparent: true, opacity: 0.35 }));
  body.position.y = -0.12 - bodyH / 2;
  seat.add(body);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.12, 0.012, 8, 32), new THREE.MeshBasicMaterial({ color: 0xd6336c }));
  ring.rotation.x = Math.PI / 2;
  seat.add(ring);
  const seatLabel = label('seat', 'seat');
  seatLabel.position.set(0, -0.18, 0);
  seat.add(seatLabel);
  roomGroup.add(seat);

  // Camera marker (on FC).
  const cam = room.camera_xy ?? [0, room.center_distance_m ?? 2];
  camPos = toThree([cam[0], cam[1], 0]);
  camPos.y = room.camera_height_m != null ? floorZ + room.camera_height_m : 0;
  const marker = new THREE.Mesh(new THREE.ConeGeometry(0.06, 0.14, 4), new THREE.MeshStandardMaterial({ color: 0xffd43b }));
  marker.position.copy(camPos);
  marker.lookAt(new THREE.Vector3(0, 0, 0));
  marker.rotateX(Math.PI / 2);
  roomGroup.add(marker);
  const camLabel = label('camera');
  camLabel.position.copy(camPos).add(new THREE.Vector3(0, 0.15, 0));
  roomGroup.add(camLabel);

  speakers = room.speakers.map((s) => ({ ...s, position: [...s.position] }));
  buildSpeakers();
  renderRows();
  renderRoomInfo();
}

function renderRoomInfo() {
  const rows = [];
  if (room.ready) {
    rows.push(['created', room.created], ['scale', `x${room.scale.toFixed(3)} (${room.scale_source})`],
      ['camera &rarr; seat', `${room.center_distance_m.toFixed(2)} m`], ['camera FOV', `${room.hfov_deg.toFixed(1)}&deg;`]);
    if (room.camera_height_m != null) {
      rows.push(['camera height', `${room.camera_height_m.toFixed(2)} m, pitch ${fmt(room.camera_pitch_deg, 1)}&deg;`]);
      rows.push(['your face height', `${room.face_height_m.toFixed(2)} m`]);
    } else {
      rows.push(['floor', 'not visible (level camera assumed)']);
    }
  } else {
    rows.push(['room', 'no setup yet']);
  }
  $('roomInfo').innerHTML = rows.map(([k, v]) => `<div>${k}</div><div>${v}</div>`).join('');
}

// ---------------------------------------------------------------- speakers
let speakers = [];
const meshes = new Map();
const lines = new Map();
let speakerGroup = new THREE.Group();
scene.add(speakerGroup);
const lineMat = new THREE.LineBasicMaterial({ color: 0xadb5bd, transparent: true, opacity: 0.6 });

function buildSpeakers() {
  transform.detach();
  disposeGroup(speakerGroup);
  speakerGroup = new THREE.Group();
  scene.add(speakerGroup);
  meshes.clear(); lines.clear();
  for (const s of speakers) {
    const geom = s.lfe ? new THREE.BoxGeometry(0.3, 0.3, 0.3) : new THREE.BoxGeometry(0.14, 0.22, 0.14);
    const mesh = new THREE.Mesh(geom, new THREE.MeshStandardMaterial({ color: COLORS[s.name] ?? 0xffffff, roughness: 0.5 }));
    mesh.userData.name = s.name;
    mesh.position.copy(toThree(s.position));
    const lab = label(s.name);
    lab.position.set(0, s.lfe ? 0.25 : 0.2, 0);
    mesh.add(lab);
    speakerGroup.add(mesh);
    meshes.set(s.name, mesh);
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), mesh.position.clone()]), lineMat);
    line.visible = $('showLines').checked;
    speakerGroup.add(line);
    lines.set(s.name, line);
  }
  if (selected) select(selected, false);
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
  if (rerender) renderRows();
  else [...$('rows').children].forEach((tr, i) => tr.classList.toggle('sel', speakers[i]?.name === name));
}

function updateSpeaker(s) {
  const mesh = meshes.get(s.name);
  mesh.position.copy(toThree(s.position));
  lines.get(s.name).geometry.setFromPoints([new THREE.Vector3(), mesh.position.clone()]);
  const ro = $(`ro-${s.name}`);
  if (ro) ro.innerHTML = azDist(s);
  ro?.parentElement.querySelectorAll('input').forEach((input) => {
    if (document.activeElement !== input) input.value = s.position[+input.dataset.k].toFixed(2);
  });
}

function azDist(s) {
  const [x, y] = s.position;
  return `${fmt((Math.atan2(x, y) * 180) / Math.PI, 0)}&deg; / ${Math.hypot(x, y).toFixed(2)} m${s.placed ? '' : ' <span class="muted">default</span>'}`;
}

function renderRows() {
  const rowsEl = $('rows');
  rowsEl.innerHTML = '';
  speakers.forEach((s, i) => {
    const tr = document.createElement('tr');
    if (s.name === selected) tr.className = 'sel';
    const fixed = FIXED.has(s.name);
    tr.innerHTML = `<td class="name"><span class="swatch" style="background:${hex(COLORS[s.name] ?? 0xffffff)}"></span>${i + 1} ${s.name}${fixed ? ' <span class="muted">(camera)</span>' : ''}</td>`
      + [0, 1, 2].map((k) => `<td><input type="number" step="0.01" data-k="${k}" value="${s.position[k].toFixed(2)}" ${fixed ? 'disabled' : ''}></td>`).join('')
      + `<td class="ro" id="ro-${s.name}">${azDist(s)}</td>`;
    tr.addEventListener('click', (e) => { if (e.target.tagName !== 'INPUT') select(s.name); });
    tr.querySelectorAll('input').forEach((input) => {
      input.addEventListener('focus', () => { if (selected !== s.name) select(s.name, false); });
      input.addEventListener('change', () => {
        const v = parseFloat(input.value);
        if (Number.isFinite(v)) { s.position[+input.dataset.k] = v; s.placed = true; updateSpeaker(s); }
      });
    });
    rowsEl.appendChild(tr);
  });
}

// ---------------------------------------------------------------- listener (live)
const listener = new THREE.Group();
const headMat = new THREE.MeshStandardMaterial({ color: 0x1098ad, transparent: true, opacity: 1 });
listener.add(new THREE.Mesh(new THREE.SphereGeometry(0.09, 24, 16), headMat));
const nose = new THREE.Mesh(new THREE.ConeGeometry(0.035, 0.12, 12), headMat);
nose.rotation.x = -Math.PI / 2; // points along -z = toward the front (yaw 0)
nose.position.z = -0.12;
listener.add(nose);
const listenerLabel = label('you', 'listener');
listenerLabel.position.set(0, 0.2, 0);
listener.add(listenerLabel);
listener.visible = false;
scene.add(listener);
const trail = [];
const trailGeom = new THREE.BufferGeometry();
const trailLine = new THREE.Line(trailGeom, new THREE.LineBasicMaterial({ color: 0x3bc9db, transparent: true, opacity: 0.7 }));
scene.add(trailLine);
let live = null;

function updateListener(s) {
  if (s.x == null) return;
  const p = toThree([s.x, s.y, s.z ?? 0]);
  listener.position.copy(p);
  listener.rotation.y = -((s.yaw_deg ?? 0) * Math.PI) / 180; // + yaw = turned right (clockwise from above)
  listener.visible = true;
  headMat.opacity = s.tracking ? 1 : 0.35;
  const last = trail[trail.length - 1];
  if (!last || last.distanceTo(p) > 0.005) {
    trail.push(p.clone());
    if (trail.length > TRAIL_LEN) trail.shift();
    trailGeom.setFromPoints(trail);
  }
  trailLine.visible = $('showTrail').checked;
}

function updateLivePanel(s) {
  const pill = $('trackPill');
  if (s.live === false) { pill.className = 'pill idle'; pill.textContent = 'no camera'; return; }
  pill.className = `pill ${s.tracking ? 'ok' : 'bad'}`;
  pill.textContent = s.tracking ? 'tracking' : (s.x == null ? 'searching' : 'face lost');
  $('modeInfo').textContent = s.mode ? `${s.mode}` : '';
  if (s.x != null) {
    $('vx').textContent = fmt(s.x); $('vy').textContent = fmt(s.y); $('vz').textContent = fmt(s.z ?? 0);
    $('vdist').textContent = `${s.distance.toFixed(2)} m`;
    $('vyaw').innerHTML = `${fmt(s.yaw_deg, 0)}&deg;`;
    $('vscore').textContent = s.score.toFixed(2);
  }
  $('kCam').textContent = `${(s.camera_fps ?? 0).toFixed(1)} fps`;
  $('kDet').textContent = `${(s.detect_fps ?? 0).toFixed(1)} fps` + (s.target_fps ? ` (target ${s.target_fps.toFixed(0)})` : '');
  $('kLat').textContent = s.latency_ms != null ? `${s.latency_ms.toFixed(1)} ms / ${s.infer_ms.toFixed(1)} ms` : '–';
  $('kCal').textContent = s.calibrated ? 'from room setup (depth)' : 'defaults - run the setup';
  $('kBridge').textContent = s.bridge ? `${s.bridge.host}:${s.bridge.port}, ${s.bridge.sent ?? 0} sent` : '–';
}

let roomVersion = null;
function updateSetup(st, version) {
  if (roomVersion !== null && version !== roomVersion) loadRoom();
  roomVersion = version;
  const pill = $('setupPill');
  pill.textContent = st.stage;
  pill.className = `pill ${st.stage === 'done' ? 'ok' : st.stage === 'error' ? 'bad' : st.stage === 'idle' ? 'idle' : 'ok'}`;
  $('setupMsg').textContent = st.message ?? '';
  $('runSetup').disabled = !['idle', 'done', 'error'].includes(st.stage);
  const log = $('setupLog');
  log.hidden = !(st.log && st.log.length);
  log.textContent = (st.log ?? []).join('\n');
  $('camNote').textContent = st.stage === 'countdown' ? `setup in ${Math.ceil(st.remaining ?? 0)} s`
    : st.stage === 'capturing' ? 'capturing...' : '';
}

const events = new EventSource('/api/events');
events.onmessage = (e) => {
  const s = JSON.parse(e.data);
  live = s;
  updateLivePanel(s);
  if (s.live === false) return;
  updateListener(s);
  updateSetup(s.setup, s.room_version);
  updateCamera(s.camera);
};

const preview = $('preview');
preview.src = '/api/video.mjpg';
preview.onerror = () => { $('camNote').textContent = 'no camera'; };

$('runSetup').addEventListener('click', async () => {
  const r = await fetch('/api/setup', { method: 'POST' });
  if (!r.ok) $('setupMsg').textContent = 'setup already running (or no camera)';
});

// ---------------------------------------------------------------- top view (2D)
const topCanvas = $('top');
const tctx = topCanvas.getContext('2d');
function drawTop() {
  const dpr = window.devicePixelRatio;
  const cw = topCanvas.clientWidth, ch = topCanvas.clientHeight;
  if (topCanvas.width !== cw * dpr) { topCanvas.width = cw * dpr; topCanvas.height = ch * dpr; }
  tctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  tctx.fillStyle = '#fff';
  tctx.fillRect(0, 0, cw, ch);

  // view: fit the map (or +-3 m) around the seat
  let ppm, ox, oy; // px per meter, canvas px of the seat
  const meta = room?.map;
  if (meta && mapImg?.complete && mapImg.naturalWidth) {
    const k = Math.min(cw / meta.size_px[0], ch / meta.size_px[1]);
    const iw = meta.size_px[0] * k, ih = meta.size_px[1] * k;
    const x0 = (cw - iw) / 2, y0 = (ch - ih) / 2;
    tctx.imageSmoothingEnabled = false;
    tctx.drawImage(mapImg, x0, y0, iw, ih);
    ppm = k / meta.m_per_px;
    ox = x0 + meta.origin_px[0] * k; oy = y0 + meta.origin_px[1] * k;
  } else {
    ppm = Math.min(cw, ch) / 6; ox = cw / 2; oy = ch / 2;
  }
  const P = (x, y) => [ox + x * ppm, oy - y * ppm];

  tctx.strokeStyle = 'rgba(0,0,0,.08)';
  tctx.lineWidth = 1;
  for (let m = -10; m <= 10; m++) {
    const [gx] = P(m, 0); const [, gy] = P(0, m);
    tctx.beginPath(); tctx.moveTo(gx, 0); tctx.lineTo(gx, ch); tctx.stroke();
    tctx.beginPath(); tctx.moveTo(0, gy); tctx.lineTo(cw, gy); tctx.stroke();
  }
  // seat
  const [sx, sy] = P(0, 0);
  tctx.strokeStyle = '#d6336c'; tctx.lineWidth = 2;
  tctx.beginPath(); tctx.arc(sx, sy, 7, 0, 2 * Math.PI); tctx.stroke();
  // speakers
  tctx.font = '600 11px -apple-system, sans-serif';
  for (const s of speakers) {
    const [x, y] = P(s.position[0], s.position[1]);
    tctx.fillStyle = hex(COLORS[s.name] ?? 0x333333);
    tctx.fillRect(x - 5, y - 5, 10, 10);
    tctx.fillStyle = '#111';
    tctx.fillText(s.name, x + 8, y + 4);
  }
  // suggested positions
  if (suggestion && $('showGhosts').checked) {
    tctx.setLineDash([3, 3]);
    for (const s of suggestion.speakers) {
      const [x, y] = P(s.position[0], s.position[1]);
      tctx.strokeStyle = hex(COLORS[s.name] ?? 0x333333);
      tctx.lineWidth = 1.5;
      tctx.strokeRect(x - 7, y - 7, 14, 14);
      const cur = speakers.find((sp) => sp.name === s.name);
      if (cur) {
        const [cx, cy] = P(cur.position[0], cur.position[1]);
        tctx.beginPath(); tctx.moveTo(cx, cy); tctx.lineTo(x, y); tctx.stroke();
      }
    }
    tctx.setLineDash([]);
  }
  // trail + listener
  if (trail.length > 1) {
    tctx.strokeStyle = 'rgba(16,152,173,.5)'; tctx.lineWidth = 2;
    tctx.beginPath();
    trail.forEach((p, i) => { const [x, y] = P(...toRoom(p)); if (i) tctx.lineTo(x, y); else tctx.moveTo(x, y); });
    tctx.stroke();
  }
  if (live?.x != null) {
    const [lx, ly] = P(live.x, live.y);
    const yaw = ((live.yaw_deg ?? 0) * Math.PI) / 180;
    tctx.fillStyle = live.tracking ? '#1098ad' : 'rgba(16,152,173,.4)';
    tctx.beginPath(); tctx.arc(lx, ly, 7, 0, 2 * Math.PI); tctx.fill();
    tctx.strokeStyle = tctx.fillStyle; tctx.lineWidth = 3;
    tctx.beginPath(); tctx.moveTo(lx, ly); tctx.lineTo(lx + Math.sin(yaw) * 18, ly - Math.cos(yaw) * 18); tctx.stroke();
    tctx.fillStyle = '#111';
    tctx.fillText(`${fmt(live.x)}, ${fmt(live.y)} m`, lx + 10, ly + 16);
  }
}

// ---------------------------------------------------------------- suggestion mode
let suggestion = null;
let ghostGroup = new THREE.Group();
scene.add(ghostGroup);

function buildGhosts() {
  disposeGroup(ghostGroup);
  ghostGroup = new THREE.Group();
  scene.add(ghostGroup);
  if (!suggestion) return;
  ghostGroup.visible = $('showGhosts').checked;
  const seat = new THREE.Vector3(0, 0, 0);
  for (const s of suggestion.speakers) {
    const color = COLORS[s.name] ?? 0xffffff;
    const geom = s.lfe ? new THREE.BoxGeometry(0.3, 0.3, 0.3) : new THREE.BoxGeometry(0.14, 0.22, 0.14);
    const p = toThree(s.position);
    const box = new THREE.Mesh(geom, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.22, depthWrite: false }));
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geom), new THREE.LineBasicMaterial({ color }));
    box.position.copy(p);
    edges.position.copy(p);
    const aim = new THREE.ArrowHelper(new THREE.Vector3().subVectors(seat, p).normalize(), p, 0.4, color, 0.09, 0.05);
    const lab = label(`${s.name} ✓`, 'ghost');
    lab.position.copy(p).add(new THREE.Vector3(0, s.lfe ? 0.3 : 0.25, 0));
    ghostGroup.add(box, edges, aim, lab);
    const cur = speakers.find((sp) => sp.name === s.name);
    if (cur && cur.position.some((v, k) => Math.abs(v - s.position[k]) > 0.02)) {
      const g = new THREE.BufferGeometry().setFromPoints([toThree(cur.position), p]);
      const line = new THREE.Line(g, new THREE.LineDashedMaterial({ color, dashSize: 0.06, gapSize: 0.04 }));
      line.computeLineDistances();
      ghostGroup.add(line);
    }
  }
}

function analysisText(a) {
  return `distance spread ${a.distance_spread_m.toFixed(2)} m, max delay ${a.max_delay_ms.toFixed(2)} ms`;
}

function renderSuggestion() {
  const table = $('suggestTable');
  const rows = $('suggestRows');
  rows.innerHTML = '';
  table.hidden = !suggestion;
  $('applyAll').disabled = !suggestion;
  if (!suggestion) return;
  for (const s of suggestion.speakers) {
    const tr = document.createElement('tr');
    const fixed = FIXED.has(s.name);
    tr.innerHTML = `<td class="name"><span class="swatch" style="background:${hex(COLORS[s.name] ?? 0xffffff)}"></span>${s.name}`
      + `${s.verified ? '' : ' <span class="badge warn" title="Not visible in the scan - check the spot is free">unverified</span>'}</td>`
      + `<td class="ro">${fmt(s.azimuth_deg, 0)}&deg;</td><td class="ro">${s.distance_m.toFixed(2)} m</td>`
      + `<td class="ro">${fmt(s.position[2])} m</td>`
      + `<td class="ro" title="point it at the seat">tilt ${fmt(s.aim_pitch_deg, 0)}&deg;</td>`
      + `<td>${fixed ? '<span class="muted">fixed</span>' : `<button data-apply="${s.name}">Apply</button>`}</td>`;
    rows.appendChild(tr);
  }
  rows.querySelectorAll('[data-apply]').forEach((b) => b.addEventListener('click', () => applySuggestion([b.dataset.apply])));

  const notes = [];
  for (const s of suggestion.speakers) for (const n of s.notes) notes.push(`<li><b>${s.name}</b>: ${n}</li>`);
  for (const n of suggestion.room) notes.push(`<li><b>room</b>: ${n}</li>`);
  $('suggestNotes').innerHTML =
    `<div><b>Centering at the seat</b> (0 = centered without DSP correction)</div>`
    + `<div>suggested: ${analysisText(suggestion.analysis)}</div>`
    + `<div class="muted">current: ${analysisText(suggestion.current_analysis)}</div>`
    + `<div style="margin-top:6px">arc radius ${suggestion.arc_radius_m.toFixed(2)} m (seat &rarr; FC) &middot; scan covers ${(suggestion.scan_coverage * 100).toFixed(0)}% of directions</div>`
    + (notes.length ? `<ul>${notes.join('')}</ul>` : '');
}

function applySuggestion(names) {
  for (const name of names) {
    const sug = suggestion.speakers.find((s) => s.name === name);
    const s = speakers.find((sp) => sp.name === name);
    if (!sug || !s || FIXED.has(name)) continue;
    s.position = [...sug.position];
    s.placed = true;
    updateSpeaker(s);
  }
  buildGhosts();
  $('suggestMsg').textContent = `applied ${names.filter((n) => !FIXED.has(n)).join(', ')} - press Save in the Speakers tab to keep it`;
}

$('runSuggest').addEventListener('click', async () => {
  $('suggestMsg').textContent = 'analyzing the scan...';
  const r = await fetch('/api/suggest');
  const data = await r.json();
  if (!r.ok) { $('suggestMsg').textContent = data.error ?? 'failed'; return; }
  suggestion = data;
  $('suggestMsg').textContent = 'Suggested positions are the outlined boxes; arrows show where to aim them.';
  buildGhosts();
  renderSuggestion();
  setView('top');
});
$('applyAll').addEventListener('click', () => applySuggestion(suggestion.speakers.map((s) => s.name)));
$('showGhosts').addEventListener('change', (e) => { ghostGroup.visible = e.target.checked; });

// ---------------------------------------------------------------- camera tab
const COMMON_MODES = [[640, 480], [1280, 720], [1280, 960], [1920, 1080], [1920, 1440]];
let camState = null;
let modesKey = '';
let controlsKey = '';

async function cameraAction(action, extra = {}) {
  const r = await fetch('/api/camera', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, ...extra }) });
  if (!r.ok) $('camStatus').textContent = (await r.json()).error ?? 'failed';
}

function updateCamera(c) {
  if (!c) return;
  camState = c;
  const st = c.settings;
  $('camNow').textContent = `camera ${st.index}: ${c.actual[0]}x${c.actual[1]} @ ${st.fps} fps`
    + (Object.keys(st.controls ?? {}).length ? ` · ${Object.entries(st.controls).map(([k, v]) => `${k} ${(+v).toFixed(1)}`).join(', ')}` : '');
  $('camStatus').textContent = c.busy ? `${c.busy}...` : (c.message || 'ready');
  for (const id of ['camList', 'camProbe', 'camWidest', 'camApply']) $(id).disabled = !!c.busy;
  const warn = $('camWarn');
  if (c.matches_setup === false) {
    warn.hidden = false;
    warn.textContent = `The camera or its view changed since the room setup (setup used camera ${c.setup_camera.index}, `
      + `${c.setup_camera.size[0]}x${c.setup_camera.size[1]}). Positions and the scan no longer fit: run Setup again.`;
  } else {
    warn.hidden = true;
  }

  // mode list: probed modes if known, else common modes
  const modes = c.capabilities?.modes ?? COMMON_MODES;
  const key = JSON.stringify(modes) + c.actual.join('x');
  if (key !== modesKey) {
    modesKey = key;
    const sel = $('camMode');
    sel.innerHTML = modes.map(([w, h]) => {
      const a = w / h;
      const tag = Math.abs(a - 4 / 3) < 0.02 ? '4:3' : Math.abs(a - 16 / 9) < 0.02 ? '16:9' : a.toFixed(2);
      return `<option value="${w}x${h}">${w}x${h} (${tag})</option>`;
    }).join('');
    sel.value = `${c.actual[0]}x${c.actual[1]}`;
    if (!sel.value && sel.options.length) sel.selectedIndex = 0;
    $('camFps').value = String(st.fps);
  }

  // control sliders for what the driver supports
  const controls = c.capabilities?.controls ?? {};
  const ckey = JSON.stringify(Object.keys(controls));
  if (ckey !== controlsKey) {
    controlsKey = ckey;
    const box = $('camControls');
    box.innerHTML = Object.entries(controls).map(([name, r]) => {
      const step = r.max - r.min > 20 ? 1 : (r.max - r.min) / 100;
      const v = st.controls?.[name] ?? r.value;
      return `<div class="ctl"><span class="muted">${name}</span>`
        + `<input type="range" data-ctl="${name}" min="${r.min}" max="${r.max}" step="${step}" value="${v}">`
        + `<span class="ro" id="ctl-${name}">${(+v).toFixed(1)}</span></div>`;
    }).join('') || (c.capabilities ? '<div class="muted">No adjustable controls exposed by this camera.</div>' : '');
    box.querySelectorAll('[data-ctl]').forEach((input) => {
      input.addEventListener('input', () => { $(`ctl-${input.dataset.ctl}`).textContent = (+input.value).toFixed(1); });
      input.addEventListener('change', () => cameraAction('control', { name: input.dataset.ctl, value: +input.value }));
    });
  }
}

async function refreshCameraGrid() {
  const { cameras } = await fetch('/api/cameras').then((r) => r.json());
  $('camGrid').innerHTML = cameras.map((cam) => `<div class="camcard ${cam.active ? 'active' : ''}" data-index="${cam.index}">`
    + `<img src="data:image/jpeg;base64,${cam.snapshot}" alt="camera ${cam.index}">`
    + `<div><b>${cam.index}</b> ${cam.name} &middot; ${cam.width}x${cam.height}${cam.active ? ' &middot; in use' : ''}</div></div>`).join('');
  $('camGrid').querySelectorAll('.camcard').forEach((card) => card.addEventListener('click', () => {
    const idx = card.dataset.index;
    if (!card.classList.contains('active')) cameraAction('open', { index: /^\d+$/.test(idx) ? +idx : idx });
  }));
  if (cameras.length <= 1) {
    $('camGrid').insertAdjacentHTML('beforeend', '<div class="muted" style="grid-column: 1 / -1">Only one camera found. '
      + 'For a phone, see "Phone as camera" below.</div>');
  }
}

async function waitIdleThen(fn) {
  for (let i = 0; i < 200; i++) {
    await new Promise((r) => setTimeout(r, 250));
    if (camState && !camState.busy) break;
  }
  fn();
}

$('camList').addEventListener('click', async () => {
  await cameraAction('list');
  await new Promise((r) => setTimeout(r, 400));
  waitIdleThen(refreshCameraGrid);
});
$('camProbe').addEventListener('click', () => cameraAction('probe'));
$('camUrlGo').addEventListener('click', () => {
  const url = $('camUrl').value.trim();
  if (!/^[a-z]+:\/\//i.test(url)) { $('camStatus').textContent = 'enter a URL like http://phone-ip:8080/video'; return; }
  cameraAction('open', { index: url });
});
$('camWidest').addEventListener('click', () => cameraAction('widest'));
$('camApply').addEventListener('click', () => {
  const [w, h] = $('camMode').value.split('x').map(Number);
  cameraAction('open', { width: w, height: h, fps: +$('camFps').value });
});

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
  if (e.shiftKey && selected && !FIXED.has(selected) && cloud) {
    const hit = raycaster.intersectObject(cloud)[0];
    if (hit) {
      const s = speakers.find((sp) => sp.name === selected);
      s.position = toRoom(hit.point).map((v) => Math.round(v * 1000) / 1000);
      s.placed = true;
      updateSpeaker(s);
      $('status').textContent = `${s.name} placed on the scan surface (not saved yet)`;
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

// ---------------------------------------------------------------- views + controls
function setView(name) {
  document.querySelectorAll('[data-view]').forEach((b) => b.classList.toggle('on', b.dataset.view === name));
  const front = toThree([0, Math.max(room?.center_distance_m ?? 2, 1), 0]);
  if (name === 'camera') {
    camera.position.copy(camPos);
    orbit.target.set(0, 0, 0);
  } else if (name === 'top') {
    camera.position.set(0, 7, front.z / 2 + 0.001);
    orbit.target.set(0, floorZ, front.z / 2);
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
$('psize').addEventListener('input', (e) => { cloudMat.size = e.target.value / 1000; });
$('showLines').addEventListener('change', (e) => { for (const l of lines.values()) l.visible = e.target.checked; });
$('showTrail').addEventListener('change', (e) => { trailLine.visible = e.target.checked; });
$('reset').addEventListener('click', async () => {
  speakers = (await fetch('/api/defaults').then((r) => r.json())).map((s) => ({ ...s, position: [...s.position] }));
  buildSpeakers();
  renderRows();
  $('status').textContent = 'reset to defaults (not saved yet)';
});
$('save').addEventListener('click', async () => {
  const r = await fetch('/api/save', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ speakers }) });
  $('status').textContent = r.ok ? `saved ${new Date().toLocaleTimeString()} - the DSP config now uses these positions` : 'save failed';
});

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  labels.setSize(window.innerWidth, window.innerHeight);
});

// ---------------------------------------------------------------- start
await loadRoom();
setView(room.ready ? 'camera' : 'free');
renderer.setAnimationLoop(() => {
  orbit.update();
  renderer.render(scene, camera);
  labels.render(scene, camera);
  drawTop();
});
