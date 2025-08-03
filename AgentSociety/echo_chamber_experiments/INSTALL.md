# Quick Installation Guide

## 🚀 Easy Setup for Any Computer

### Method 1: Direct Download & Run (Recommended)

1. **Download the project**:
   ```bash
   git clone https://github.com/your-username/AgentSociety-Project.git
   cd AgentSociety-Project/echo_chamber_experiments
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run your first experiment**:
   ```bash
   python run_experiment.py run --experiment basic_polarization
   ```

That's it! 🎉

### Method 2: Full Installation (For Developers)

1. **Install as package**:
   ```bash
   cd AgentSociety-Project/echo_chamber_experiments
   pip install -e .
   ```

2. **Use the command-line tool**:
   ```bash
   echo-chamber run --experiment basic_polarization
   ```

---

## 🖥️ Platform Compatibility

✅ **Windows** - Works perfectly  
✅ **macOS** - Intel and Apple Silicon  
✅ **Linux** - All distributions  

**No special hardware needed** - Pure Python implementation!

---

## 🎯 Quick Test

Test your installation:
```bash
python examples/simple_example.py
```

If you see experiment results, you're ready to go! 🚀

---

## 📊 View Results

After running experiments:
- Check `results/` folder for data and reports
- Open `interactive_dashboard.html` in your browser
- Import CSV files into Excel/R/Python for analysis

---

## 🆘 Troubleshooting

**Import errors?**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python run_experiment.py run --experiment basic_polarization
```

**Missing modules?**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Need help?** Check the main README.md or open an issue!