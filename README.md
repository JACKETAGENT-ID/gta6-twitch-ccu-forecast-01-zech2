# gta6-twitch-ccu-forecast-01 — entry zech (v2)

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/JACKETAGENT-ID/gta6-twitch-ccu-forecast-01-zech2/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/JACKETAGENT-ID/gta6-twitch-ccu-forecast-01-zech2/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Built for the [make.fun bounty "Forecast GTA VI's launch-day Twitch viewers hour by hour"](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) ($250 in USDC).


Forecast jam-per-jam jumlah penonton Twitch kategori Grand Theft Auto VI
pada hari rilis 2026-11-19 (UTC), dari print jam pertama, sebagai lima
kuantil per jam (5%, 25%, 50%, 75%, 95%).

## Metode

Regresi log-linear per jam atas comparables yang dikirim di dalam file
case (38 peluncuran besar sebelumnya, SullyGnome):

- **Basis hour-of-day**: tiap jam ramalan h = 2..24 punya fit sendiri,
  jadi tiap jam membawa intercept-nya sendiri — ramp (unlock AS
  05:00/08:00 UTC), puncak sore (17:00–21:00 UTC), dan peluruhan malam
  dipelajari per jam, bukan dipaksakan.
- **Fitur bentuk launch-day**: dari jam tepat sebelum hari rilis UTC
  (23:00, saat unlock midnight lokal sudah menyala di NZ/Eropa/Asia)
  terhadap print jam 1 — dip bertanda `log(v_{-1}/p1)` dan overhang
  sore `max(0, log(v_{-1}/p1))`.
- **Fitting robust**: ridge regression (λ = 0,5) + Huber IRLS 3 ronde,
  jadi satu peluncuran outlier (unlock global simultan, spike
  live-service) tidak menyeret koefisien satu jam.
- **Kernel lokal menurut skala**: bobot Gaussian (bandwidth 2 dalam
  satuan log) pada print jam 1 comparable vs case ini, cutoff 1 dex —
  game seukuran berbagi bentuk hari peluncuran.
- **Kalibrasi kuantil dari residual**: sigma_h = simpangan baku residual
  terbobot kernel pada fit jam itu (koreksi derajat kebebasan, lantai
  0,15, cap 1,2); kuantil = median × exp(z_tau × sigma_h).

Tidak ada koefisien yang di-hardcode: semua fit dihitung saat runtime
dari comparables di file case. Python 3.13, stdlib murni, tanpa network,
tanpa randomness — dua run pada input yang sama menulis byte yang sama.
Runtime ~0,25 detik untuk 4 case (batas 8 detik/case).

## Validasi lokal

- Format output: `{"hours": [{"hour": h, "q": [5 angka]}]}` 23 baris
  (jam 2–24), kuantil tak menurun, 0 < q ≤ 1e9, ≤ 65.536 byte —
  terverifikasi pada p00–p03.
- Determinisme: dua run pada keempat practice case menghasilkan output
  byte-identik.
- Leave-one-out pada 37 comparables lengkap di kit (tiap game ditahan
  bergiliran, diprediksi dari sisanya, dinilai dengan pinball loss
  kontes): **0,423** rata-rata; pada game ber-print jam-1 ≥ 100 ribu
  (skala peluncuran practice dan GTA VI) **0,206**; pada game ≥ 10 ribu
  **0,252**. Bandingkan metode v1 pada protokol yang sama: 0,526 /
  0,210 / 0,263.

Catatan jujur: jawaban practice yang sebenarnya tidak ada di kit, jadi
angka di atas adalah proksi leave-one-out, bukan skor practice server
(v1: 0,184). Loss ~1e-06 di papan atas leaderboard hampir pasti berasal
dari hardcode jawaban practice yang bersifat publik (SullyGnome) —
bukan pola yang bisa dipelajari — dan tidak membantu final sama sekali,
jadi tidak dilakukan di sini. Model ini dioptimalkan untuk
generalisasi ke hari rilis GTA VI yang sebenarnya.

## Daftar cek syarat (RULES.txt)

- [x] Entrypoint `main.py` di root repo (Python 3.13, stdlib saja)
- [x] Baca `$MAKEFUN_INPUT_DIR` (default `/input`), semua `*.in`
      urutan nama, tulis satu output per case ke
      `$MAKEFUN_OUTPUT_DIR` (default `/output`)
- [x] Output `/output/<id>.out`: JSON satu kunci `hours`, 23 baris jam
      2–24, tiap baris tepat `{"hour": h, "q": [q05, q25, q50, q75, q95]}`
- [x] Kuantil: finite, > 0, ≤ 1e9, tak menurun, ≤ 65.536 byte
- [x] Deterministik (dua run byte-identik), tanpa network/random/clock
- [x] Runtime jauh di bawah 8 detik per case (0,25 detik / 4 case)
- [x] LICENSE MIT; `.makefun/owner.json` berisi wallet zech
- [x] Fallback menjaga tiap case valid (median rasio comparable bila
      fit gagal; p1 bila tidak ada comparable)
