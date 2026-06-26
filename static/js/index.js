// Copy BibTeX to clipboard
document.addEventListener('DOMContentLoaded', function () {
  var btn = document.getElementById('copyBib');
  var pre = document.getElementById('bibText');
  if (!btn || !pre) return;

  btn.addEventListener('click', function () {
    var text = pre.innerText;
    var done = function () {
      var old = btn.textContent;
      btn.textContent = 'Copied!';
      setTimeout(function () { btn.textContent = old; }, 1500);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done).catch(fallback);
    } else {
      fallback();
    }
    function fallback() {
      var ta = document.createElement('textarea');
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand('copy'); done(); } catch (e) {}
      document.body.removeChild(ta);
    }
  });
});
