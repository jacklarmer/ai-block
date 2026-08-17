const http = require("http"), fs = require("fs"), path = require("path");
const MIME = { ".html": "text/html", ".js": "text/javascript", ".png": "image/png", ".jpg": "image/jpeg", ".onnx": "application/octet-stream", ".wasm": "application/wasm" };
const root = path.resolve(__dirname, "..");
http.createServer((req, res) => {
  let p = path.normalize(path.join(root, decodeURIComponent(req.url.split("?")[0])));
  if (!p.startsWith(root)) { res.writeHead(403); return res.end(); }
  try { const s = fs.statSync(p); if (s.isDirectory()) p = path.join(p, "index.html"); }
  catch (e) { res.writeHead(404, { "Content-Type": "text/plain" }); return res.end("nf"); }
  const ext = path.extname(p);
  res.writeHead(200, { "Content-Type": MIME[ext] || "application/octet-stream", "Access-Control-Allow-Origin": "*" });
  fs.createReadStream(p).pipe(res);
}).listen(8899, "127.0.0.1", () => console.log("qa server on 8899, root=" + root));
