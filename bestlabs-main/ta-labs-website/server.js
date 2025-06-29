const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { exec } = require('child_process');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(cors());
app.use(bodyParser.json());

app.post('/run', async (req, res) => {
  const { code, language } = req.body;

  const tempDir = './temp';
  if (!fs.existsSync(tempDir)) fs.mkdirSync(tempDir);

  let fileName, command;

  if (language === 'python') {
    fileName = path.join(tempDir, `temp.py`);
    fs.writeFileSync(fileName, code);
    command = `python3 ${fileName}`;
  } else if (language === 'cpp') {
    fileName = path.join(tempDir, `temp.cpp`);
    const exeName = path.join(tempDir, `temp.out`);
    fs.writeFileSync(fileName, code);
    command = `g++ ${fileName} -o ${exeName} && ${exeName}`;
  } else if (language === 'javascript') {
    fileName = path.join(tempDir, `temp.js`);
    fs.writeFileSync(fileName, code);
    command = `node ${fileName}`;
  } else {
    return res.json({ error: 'Unsupported language' });
  }

  exec(command, { timeout: 5000 }, (err, stdout, stderr) => {
    if (err) {
      return res.json({ error: stderr || err.message });
    }
    res.json({ output: stdout });
  });
});

app.listen(5050, () => console.log('Backend running on http://localhost:5000'));
