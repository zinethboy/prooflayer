/**
 * ProofLayer SDK v0.1.0
 * Embed human verification in any web app
 * 
 * Usage:
 *   const proof = new ProofLayer({ apiKey: 'your_key', apiUrl: 'https://prooflayer.up.railway.app' });
 *   proof.observe(document.getElementById('myInput'));
 *   
 *   // Later...
 *   const result = await proof.verify();
 *   if (result.is_human > 0.7) {
 *     showVerifiedBadge(result.proof_hash);
 *   }
 */

class ProofLayer {
    constructor(options = {}) {
        this.apiKey = options.apiKey || '';
        this.apiUrl = options.apiUrl || 'https://prooflayer.up.railway.app';
        this.sessionId = null;
        this.keystrokes = [];
        this.sessionStart = null;
        this.targets = new Map(); // element -> {sessionId, keystrokes}
        this.onVerify = options.onVerify || null;
        this.onTyping = options.onTyping || null;
    }

    // ─── SESSION MANAGEMENT ────────────────────────────────

    async startSession() {
        const res = await fetch(`${this.apiUrl}/api/v1/session/start`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(this.apiKey && { 'X-API-Key': this.apiKey })
            }
        });
        const data = await res.json();
        this.sessionId = data.session_id;
        this.sessionStart = Date.now();
        this.keystrokes = [];
        return data.session_id;
    }

    // ─── OBSERVE AN INPUT ELEMENT ──────────────────────────

    observe(element, options = {}) {
        if (typeof element === 'string') {
            element = document.querySelector(element);
        }
        if (!element) {
            console.error('ProofLayer: Element not found');
            return this;
        }

        const config = {
            minLength: options.minLength || 20,
            autoVerify: options.autoVerify || false,
            verifyThreshold: options.verifyThreshold || 0.7,
            ...options
        };

        // Start session for this element
        this.startSession().then(sid => {
            this.targets.set(element, {
                sessionId: sid,
                keystrokes: [],
                sessionStart: Date.now(),
                config,
                pasteDetected: false
            });
        });

        // Capture keystrokes
        element.addEventListener('keydown', (e) => this._onKeyDown(element, e));
        element.addEventListener('keyup', (e) => this._onKeyUp(element, e));
        element.addEventListener('paste', (e) => this._onPaste(element, e));

        // Auto-verify on blur if enabled
        if (config.autoVerify) {
            element.addEventListener('blur', () => this.verify(element));
        }

        return this;
    }

    _onKeyDown(element, e) {
        const target = this.targets.get(element);
        if (!target) return;

        const now = Date.now() - target.sessionStart;
        target.keystrokes.push({
            key: e.key,
            timestamp: now,
            type: 'down'
        });

        if (this.onTyping) {
            this.onTyping({ element, keystrokes: target.keystrokes.length });
        }
    }

    _onKeyUp(element, e) {
        const target = this.targets.get(element);
        if (!target) return;

        const now = Date.now() - target.sessionStart;
        target.keystrokes.push({
            key: e.key,
            timestamp: now,
            type: 'up'
        });
    }

    _onPaste(element, e) {
        const target = this.targets.get(element);
        if (!target) return;
        target.pasteDetected = true;

        const now = Date.now() - target.sessionStart;
        target.keystrokes.push({
            key: 'Paste',
            timestamp: now,
            type: 'paste'
        });
    }

    // ─── VERIFY ────────────────────────────────────────────

    async verify(element) {
        if (element && typeof element === 'string') {
            element = document.querySelector(element);
        }
        
        // If no element specified, use first observed
        if (!element) {
            const entries = Array.from(this.targets.entries());
            if (entries.length === 0) {
                throw new Error('No elements being observed');
            }
            element = entries[0][0];
        }

        const target = this.targets.get(element);
        if (!target) {
            throw new Error('Element not being observed');
        }

        const text = element.value || element.textContent || '';
        if (text.length < target.config.minLength) {
            return {
                success: false,
                error: `Text too short. Minimum ${target.config.minLength} characters.`
            };
        }

        try {
            const res = await fetch(`${this.apiUrl}/api/v1/session/${target.sessionId}/keystrokes`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(this.apiKey && { 'X-API-Key': this.apiKey })
                },
                body: JSON.stringify({
                    session_id: target.sessionId,
                    text: text,
                    events: target.keystrokes
                })
            });

            const result = await res.json();
            
            // Reset for next verification
            target.keystrokes = [];
            target.pasteDetected = false;

            if (this.onVerify) {
                this.onVerify(result);
            }

            return {
                success: true,
                isHuman: result.is_human,
                confidence: result.confidence,
                proofHash: result.proof_hash,
                proofUrl: `${this.apiUrl}/api/v1/verify/${result.proof_hash}`,
                pasteDetected: target.pasteDetected,
                ...result
            };

        } catch (e) {
            return {
                success: false,
                error: e.message
            };
        }
    }

    // ─── QUICK VERIFY (one-shot) ───────────────────────────

    static async quickVerify(text, keystrokes, options = {}) {
        const sdk = new ProofLayer(options);
        await sdk.startSession();
        
        // Simulate the keystrokes into the session
        const res = await fetch(`${sdk.apiUrl}/api/v1/session/${sdk.sessionId}/keystrokes`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(sdk.apiKey && { 'X-API-Key': sdk.apiKey })
            },
            body: JSON.stringify({
                session_id: sdk.sessionId,
                text: text,
                events: keystrokes
            })
        });

        return await res.json();
    }

    // ─── UTILITIES ─────────────────────────────────────────

    getBadgeHtml(result) {
        const isHuman = result.is_human > 0.5;
        const color = isHuman ? '#00f5d4' : '#ff006e';
        const icon = isHuman ? '✓' : '✗';
        const text = isHuman ? 'Verified Human' : 'Unverified';
        
        return `
            <span style="
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 6px 14px;
                border-radius: 20px;
                font-size: 13px;
                font-weight: 600;
                font-family: sans-serif;
                background: ${color}15;
                color: ${color};
                border: 1px solid ${color}40;
            ">
                ${icon} ${text}
            </span>
        `;
    }

    destroy(element) {
        if (element) {
            this.targets.delete(element);
        } else {
            this.targets.clear();
        }
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ProofLayer };
}
