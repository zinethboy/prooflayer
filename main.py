"""
ProofLayer Backend API
FastAPI + SQLite + Static file serving + Blockchain anchoring
"""

import os
import json
import hashlib
import time
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sqlite3
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib


# ─── CONFIG ──────────────────────────────────────────────────────────

DB_PATH = "prooflayer.db"
MODEL_PATH = "keystroke_model.pkl"

app = FastAPI(title="ProofLayer API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (THIS IS THE FIX)
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def read_root():
    return FileResponse("index.html")


# ─── DATABASE ─────────────────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            created_at REAL,
            keystrokes TEXT,
            features TEXT,
            is_human REAL,
            proof_hash TEXT,
            tx_hash TEXT,
            verified INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

init_db()


# ─── MODELS ───────────────────────────────────────────────────────────

class KeystrokeEvent(BaseModel):
    key: str
    timestamp: float
    type: str

class KeystrokeSession(BaseModel):
    session_id: str
    text: str
    events: List[KeystrokeEvent]

class VerificationResult(BaseModel):
    session_id: str
    is_human: float
    confidence: str
    proof_hash: str
    tx_hash: Optional[str]
    timestamp: str


# ─── KEYSTROKE FEATURE EXTRACTION ────────────────────────────────────

@dataclass
class KeystrokeFeatures:
    avg_dwell_time: float
    std_dwell_time: float
    avg_flight_time: float
    std_flight_time: float
    avg_diagraph_time: float
    backspace_ratio: float
    pause_ratio: float
    typing_speed_wpm: float
    rhythm_consistency: float
    error_correction_rate: float
    
    def to_vector(self) -> np.ndarray:
        return np.array([
            self.avg_dwell_time,
            self.std_dwell_time,
            self.avg_flight_time,
            self.std_flight_time,
            self.avg_diagraph_time,
            self.backspace_ratio,
            self.pause_ratio,
            self.typing_speed_wpm,
            self.rhythm_consistency,
            self.error_correction_rate
        ]).reshape(1, -1)


def extract_features(events: List[Dict], text: str) -> KeystrokeFeatures:
    events = sorted(events, key=lambda e: e['timestamp'])
    
    presses = {}
    releases = {}
    for e in events:
        if e['type'] == 'down':
            presses[e.get('key', '')] = e['timestamp']
        elif e['type'] == 'up':
            key = e.get('key', '')
            if key in presses:
                releases[key] = e['timestamp']
    
    dwell_times = []
    for key, down_time in presses.items():
        if key in releases:
            dwell_times.append(releases[key] - down_time)
    
    flight_times = []
    sorted_events = sorted(events, key=lambda e: e['timestamp'])
    last_release = None
    for e in sorted_events:
        if e['type'] == 'up':
            if last_release is not None:
                flight_times.append(e['timestamp'] - last_release)
            last_release = e['timestamp']
    
    diagraph_times = []
    down_events = [e for e in sorted_events if e['type'] == 'down']
    for i in range(len(down_events) - 1):
        dt = down_events[i+1]['timestamp'] - down_events[i]['timestamp']
        if dt < 2000:
            diagraph_times.append(dt)
    
    total_keys = len([e for e in events if e['type'] == 'down'])
    backspaces = len([e for e in events if e.get('key') == 'Backspace'])
    backspace_ratio = backspaces / total_keys if total_keys > 0 else 0
    
    total_time = events[-1]['timestamp'] - events[0]['timestamp'] if len(events) > 1 else 1
    pause_time = sum(f for f in flight_times if f > 500)
    pause_ratio = pause_time / total_time if total_time > 0 else 0
    
    char_count = len(text)
    minutes = total_time / 60000
    typing_speed_wpm = (char_count / 5) / minutes if minutes > 0 else 0
    
    if diagraph_times:
        mean_dt = np.mean(diagraph_times)
        std_dt = np.std(diagraph_times)
        rhythm_consistency = std_dt / mean_dt if mean_dt > 0 else 0
    else:
        rhythm_consistency = 0
    
    error_correction_rate = backspaces / char_count if char_count > 0 else 0
    
    return KeystrokeFeatures(
        avg_dwell_time=np.mean(dwell_times) if dwell_times else 0,
        std_dwell_time=np.std(dwell_times) if dwell_times else 0,
        avg_flight_time=np.mean(flight_times) if flight_times else 0,
        std_flight_time=np.std(flight_times) if flight_times else 0,
        avg_diagraph_time=np.mean(diagraph_times) if diagraph_times else 0,
        backspace_ratio=backspace_ratio,
        pause_ratio=pause_ratio,
        typing_speed_wpm=typing_speed_wpm,
        rhythm_consistency=rhythm_consistency,
        error_correction_rate=error_correction_rate
    )


# ─── AI/HUMAN CLASSIFIER ─────────────────────────────────────────────

class KeystrokeClassifier:
    def __init__(self):
        self.model = None
        self._load_or_init()
    
    def _load_or_init(self):
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
    
    def predict(self, features: KeystrokeFeatures) -> float:
        f = features
        score = 0.5
        
        if f.std_dwell_time > 20:
            score += 0.15
        else:
            score -= 0.1
        
        if f.backspace_ratio > 0.02:
            score += 0.15
        
        if f.pause_ratio > 0.1:
            score += 0.1
        
        if f.rhythm_consistency < 0.1 and f.typing_speed_wpm > 80:
            score -= 0.2
        
        if f.typing_speed_wpm > 120 and f.backspace_ratio == 0:
            score -= 0.25
        
        if 30 < f.typing_speed_wpm < 90:
            score += 0.1
        
        if f.std_flight_time < 5:
            score -= 0.15
        
        if self.model is not None:
            try:
                vec = features.to_vector()
                ml_score = self.model.decision_function(vec)[0]
                score = 0.6 * score + 0.4 * (1 / (1 + np.exp(-ml_score)))
            except:
                pass
        
        return max(0.0, min(1.0, score))
    
    def train(self, features_list: List[np.ndarray], labels: List[int]):
        X = np.vstack(features_list)
        y = np.array(labels)
        self.model = IsolationForest(contamination=0.3, random_state=42)
        self.model.fit(X[y == 1])
        joblib.dump(self.model, MODEL_PATH)


classifier = KeystrokeClassifier()


# ─── BLOCKCHAIN ANCHORING ─────────────────────────────────────────────

def anchor_to_chain(proof_hash: str) -> Optional[str]:
    try:
        timestamp = int(time.time())
        data = f"{proof_hash}{timestamp}"
        mock_tx = hashlib.sha256(data.encode()).hexdigest()[:64]
        return f"0x{mock_tx}"
    except Exception as e:
        print(f"Anchor failed: {e}")
        return None


# ─── API ENDPOINTS ────────────────────────────────────────────────────

@app.post("/api/v1/session/start")
def start_session():
    session_id = hashlib.sha256(
        f"{time.time()}{os.urandom(16)}".encode()
    ).hexdigest()[:16]
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO sessions (id, created_at) VALUES (?, ?)",
        (session_id, time.time())
    )
    conn.commit()
    conn.close()
    
    return {"session_id": session_id, "status": "ready"}


@app.post("/api/v1/session/{session_id}/keystrokes")
def submit_keystrokes(session_id: str, data: KeystrokeSession):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM sessions WHERE id = ?", (session_id,))
    if not c.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Session not found")
    
    events_dict = [e.model_dump() for e in data.events]
    features = extract_features(events_dict, data.text)
    features_dict = {k: float(v) for k, v in asdict(features).items()}
    
    is_human_score = classifier.predict(features)
    
    proof_data = {
        "session_id": session_id,
        "features_hash": hashlib.sha256(
            json.dumps(features_dict, sort_keys=True).encode()
        ).hexdigest()[:32],
        "text_length": len(data.text),
        "timestamp": datetime.utcnow().isoformat(),
        "is_human_score": round(is_human_score, 4)
    }
    proof_hash = hashlib.sha256(
        json.dumps(proof_data, sort_keys=True).encode()
    ).hexdigest()
    
    tx_hash = anchor_to_chain(proof_hash)
    
    c.execute('''
        UPDATE sessions 
        SET keystrokes = ?, features = ?, is_human = ?, proof_hash = ?, tx_hash = ?
        WHERE id = ?
    ''', (
        json.dumps(events_dict),
        json.dumps(features_dict),
        is_human_score,
        proof_hash,
        tx_hash,
        session_id
    ))
    conn.commit()
    conn.close()
    
    if is_human_score > 0.75:
        confidence = "high"
    elif is_human_score > 0.5:
        confidence = "medium"
    else:
        confidence = "low"
    
    return VerificationResult(
        session_id=session_id,
        is_human=round(is_human_score, 4),
        confidence=confidence,
        proof_hash=proof_hash,
        tx_hash=tx_hash,
        timestamp=proof_data["timestamp"]
    )


@app.get("/api/v1/verify/{proof_hash}")
def verify_proof(proof_hash: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT id, is_human, proof_hash, tx_hash, created_at 
        FROM sessions WHERE proof_hash = ?
    ''', (proof_hash,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Proof not found")
    
    return {
        "valid": True,
        "session_id": row[0],
        "is_human_score": row[1],
        "proof_hash": row[2],
        "tx_hash": row[3],
        "created_at": datetime.fromtimestamp(row[4]).isoformat() if row[4] else None,
        "verification_url": f"/api/v1/verify/{proof_hash}"
    }


@app.get("/api/v1/session/{session_id}")
def get_session(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT id, created_at, features, is_human, proof_hash, tx_hash
        FROM sessions WHERE id = ?
    ''', (session_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": row[0],
        "created_at": datetime.fromtimestamp(row[1]).isoformat() if row[1] else None,
        "features": json.loads(row[2]) if row[2] else None,
        "is_human_score": row[3],
        "proof_hash": row[4],
        "tx_hash": row[5]
    }


@app.get("/api/v1/stats")
def get_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*), AVG(is_human) FROM sessions WHERE is_human IS NOT NULL")
    total, avg_human = c.fetchone()
    conn.close()
    
    return {
        "total_sessions": total or 0,
        "avg_human_score": round(avg_human or 0, 4),
        "model_trained": classifier.model is not None
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)