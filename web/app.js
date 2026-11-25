const $ = id => document.getElementById(id);
const el = (tag,text) => Object.assign(document.createElement(tag),{textContent:text});
let db, stream, busy=false, revision=0, photoURL, frozen=false, timer;
const selected=()=>Array.from($("models").querySelectorAll("input:checked"),i=>i.value);
const pct=v=>v==null?"—":(100*v).toLocaleString("id-ID",{maximumFractionDigits:3})+"%";
function row(parent,values){const tr=el("tr","");values.forEach(v=>tr.append(el("td",v)));parent.append(tr);}
function renderDataset(dataset) {
  const splits = [
    ["train", "Latih", "Melatih model"],
    ["valid", "Validasi", "Memantau training dan memilih bobot"],
    ["test", "Uji asli", "Mengukur hasil pada dataset bawaan"],
    ["personal", "Uji mandiri", "Mengukur hasil pada foto sendiri"],
  ];
  const format = new Intl.NumberFormat("id-ID");
  for (const [key, label, purpose] of splits) {
    const split = dataset.splits[key];
    const present = Object.values(split.class_instances).filter(count => count > 0).length;
    row($("dataset-splits"), [label, purpose, format.format(split.images),
      format.format(split.instances), present]);
  }

  const categories = dataset.categories;
  const train = dataset.splits.train.class_instances;
  const present = categories.filter(category => train[category.id] > 0).length;
  $("dataset-summary").textContent =
    "Ada " + categories.length + " kelas dalam konfigurasi model. " + present +
    " kelas punya contoh di data latih. Uji mandiri berisi 30 foto laptop dan 34 foto bantal.";
  $("category-heading").textContent = "Lihat " + categories.length + " kelas dan jumlah anotasinya";

  const comparison = new Set(dataset.comparison_ids);
  const ordered = [...categories].sort((a, b) =>
    Number(comparison.has(b.id)) - Number(comparison.has(a.id)) || a.name.localeCompare(b.name));
  for (const category of ordered) {
    const name = comparison.has(category.id) ? category.name + " · dibandingkan di AP" : category.name;
    const counts = splits.map(([key]) => format.format(dataset.splits[key].class_instances[category.id] ?? 0));
    row($("dataset-classes"), [name, ...counts]);
    if (comparison.has(category.id)) $("dataset-classes").lastElementChild.classList.add("ap-class");
  }
}
function reset(){
  revision++; clearTimeout(timer);
  if(!db)return;
  for(const id of ["cards","stats","classes"]) $(id).replaceChildren();
  for(const m of selected()){
    const s=db.stats[m+"_"+$("size").value],name=db.models[m];
    const card=el("article","");card.className="card";const area=el("div",s.ready?"Belum ada prediksi":"Checkpoint tidak ditemukan");
    area.className="result";area.id="result-"+m;
    const info=el("p","");info.id="info-"+m;card.append(el("h3",name),area,info);$("cards").append(card);
    row($("stats"),[name,pct(s.test.AP),pct(s.personal.AP),s.delta.toFixed(2)+" poin",pct(s.personal.AP50),pct(s.personal.AP75),s.minutes.toFixed(1)+" mnt",s.vram.toFixed(2)+" GiB"]);
    row($("classes"),[name,...["laptop","pillow"].flatMap(k=>[pct(s.test.per_class_AP[k]),pct(s.personal.per_class_AP[k])])]);
  }
  $("status").textContent="";
  schedule();
}
function schedule(){clearTimeout(timer);if(!busy&&stream&&!frozen&&$("live").checked&&selected().length)timer=setTimeout(compare,180);}
function grab(){
  const v=stream&&!frozen?$("video"):$("photo");
  const w=v.videoWidth||v.naturalWidth,h=v.videoHeight||v.naturalHeight;
  if(v.hidden||!w||!h)throw Error("Pilih foto atau tunggu webcam siap.");
  const c=document.createElement("canvas"),scale=Math.min(1,1440/Math.max(w,h));c.width=Math.max(1,Math.round(w*scale));c.height=Math.max(1,Math.round(h*scale));c.getContext("2d").drawImage(v,0,0,c.width,c.height);return c;
}
async function compare(){
  if(busy||!db)return;
  const ms=selected(),size=$("size").value,conf=$("conf").value,rev=revision;
  if(!ms.length||!$("conf").checkValidity()||!conf){$("status").textContent="Pilih model dan threshold antara 0,001 dan 1.";return;}
  busy=true;$("run").disabled=true;
  try{
    const frame=grab();const blob=await new Promise(resolve=>frame.toBlob(resolve,"image/jpeg",.93));let success=0;
    for(const m of ms){
      if(rev!==revision)break;
      if(!db.stats[m+"_"+size].ready)continue;
      $("status").textContent="Memproses "+db.models[m]+"…";
      const area=$("result-"+m),info=$("info-"+m);
      try{
        const res=await fetch("/api/infer?"+new URLSearchParams({model:m,size,conf}),{method:"POST",headers:{"Content-Type":"image/jpeg"},body:blob});
        const data=await res.json();if(!res.ok)throw Error(data.error);
        if(rev!==revision)break;
        const img=new Image();img.alt="Mask "+db.models[m];img.src="data:image/jpeg;base64,"+data.image;
        area.replaceChildren(img);info.textContent=data.count+" objek · "+data.ms+" ms · "+data.device;success++;
      }catch(e){if(rev===revision){area.textContent="Inferensi gagal";info.textContent=e.message;}}
    }
    if(rev===revision){$("status").textContent=success+" dari "+ms.length+" model selesai pada frame yang sama.";if(!success)$("live").checked=false;}
  }catch(e){$("status").textContent=e.message;$("live").checked=false;}
  finally{busy=false;$("run").disabled=false;schedule();}
}
function stop(){
  stream?.getTracks().forEach(t=>t.stop());stream=null;
  $("video").srcObject=null;$("video").hidden=true;$("live").checked=false;$("live").disabled=true;
  $("cam").textContent="Nyalakan webcam";$("freeze").hidden=true;frozen=false;clearTimeout(timer);
}
function clearPhoto(){if(photoURL)URL.revokeObjectURL(photoURL);photoURL=null;$("photo").removeAttribute("src");$("photo").hidden=true;}
$("file").onchange=async e=>{
  const f=e.target.files[0];if(!f)return;stop();clearPhoto();reset();photoURL=URL.createObjectURL(f);
  const img=$("photo");img.src=photoURL;
  try{await img.decode();img.hidden=false;$("empty").hidden=true;}catch{$("empty").hidden=false;$("status").textContent="Foto tidak terbaca. Gunakan JPEG atau PNG.";}
  e.target.value="";
};
$("cam").onclick=async()=>{
  if(stream){stop();clearPhoto();$("empty").hidden=false;reset();return;}
  try{stream=await navigator.mediaDevices.getUserMedia({video:true,audio:false});
    clearPhoto();const v=$("video");v.srcObject=stream;await v.play();v.hidden=false;$("empty").hidden=true;
    $("live").disabled=false;$("freeze").hidden=false;$("freeze").textContent="Bekukan frame";$("cam").textContent="Matikan webcam";reset();
  }catch(e){stop();$("empty").hidden=!$("photo").hidden;$("status").textContent="Webcam gagal dibuka: "+e.message;}
};
$("freeze").onclick=async()=>{
  try{if(!frozen){const frame=grab();$("photo").src=frame.toDataURL("image/png");await $("photo").decode();$("photo").hidden=false;$("video").hidden=true;frozen=true;}
    else{frozen=false;$("photo").hidden=true;$("video").hidden=false;}
    $("freeze").textContent=frozen?"Lanjutkan webcam":"Bekukan frame";$("live").disabled=frozen;reset();
  }catch(e){$("status").textContent=e.message;}
};
$("run").onclick=compare;$("live").onchange=schedule;
$("size").onchange=reset;$("conf").oninput=reset;
window.addEventListener("pagehide",stop);
fetch("/api/models").then(r=>r.json()).then(data=>{db=data;renderDataset(db.dataset);for(const [key,name] of Object.entries(db.models)){
  const l=el("label",""),i=el("input","");i.type="checkbox";i.value=key;i.checked=true;i.onchange=reset;l.append(i,document.createTextNode(" "+name));$("models").append(l);
}reset();}).catch(e=>{$("status").textContent="Data evaluasi gagal dibaca: "+e.message;});
