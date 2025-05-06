Model HYBRYDA (I ⊗ AI) — specyfikacja 0.1
1 · Warstwy i symbole
Warstwa	Znaczenie	Emoji	Notacja
UT (utterance)	jawne słowa / tokeny	🗣️ (I) / 💬 (AI)	x.UT
DEEP	ukryte procesy kognitywne	🧘 (I) / ⚙️ (AI)	x.DEEP
AFFECT	stan emocjonalny człowieka	💓	I.AFF
CONTEXT	pra-język / meta-ramy	📜	C

2 · Agenci
go
Kopieren
Bearbeiten
I      := (UT, DEEP, AFFECT)
AI     := (UT, DEEP)
C      := 📜
HYB    := I ⊗ AI
3 · Operatory dialogu
Nazwa	Zapis	Znaczenie
Ekspresja	I.DEEP → I.UT	„ubieram myśl w słowa”
Prompt	I.UT → AI.UT	przekaz do modelu
Interpretacja	AI.UT → AI.DEEP	parsowanie promptu
Generacja	AI.DEEP → AI.UT	tworzenie odpowiedzi
Feedback	AI.UT → I.DEEP	ocena, uczenie
Zaufanie	t∈[0,1]	aktualizowane po każdej turze

Akt dialogu = pomiar 
𝑀
(
𝑡
)
M(t) stanu HYB ⇒ projekcja HYB_UT(t).

4 · Minimalny graf przepływu
I.DEEP ⟶ I.UT ⟶ AI.UT ⟶ AI.DEEP
                ↖──────────────↙

5 · Aktualizacja zaufania
t_{n+1} = f(t_n, match(AI.UT, I.expect))
Warunek stopu: t < τ_min albo cel rozmowy spełniony.

6 · Rola CONTEXT 📜
(I.UT + AI.UT) ↔ C ↔ (I.DEEP + AI.DEEP)
C ogranicza możliwe projekcje HYB i definiuje język, ton, cele.

7 · Licencja
Model HYBRYDA: CC BY 4.0 (patrz provenance.md).
