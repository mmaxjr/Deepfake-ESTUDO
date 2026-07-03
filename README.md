# Ferramenta educacional de face-swap / deepfake

Pipeline em Python para estudar como funciona a tecnologia de face-swap
(detecção/alinhamento facial, autoencoder com encoder compartilhado e decoder
por identidade, treino, e recomposição em vídeo). Feito para fins de
estudo/entretenimento pessoal.

## ⚠️ Regra de uso (aplicada em código, não só aqui)

A identidade sintetizada (o rosto que o modelo aprende a reconstruir) **só
pode vir de fotos/vídeo do próprio operador da ferramenta, ou de alguém que
deu consentimento explícito e documentado**. Datasets de pesquisa
(FaceForensics++, Celeb-DF, DFDC) só podem ser usados como **vídeo de cena**
(a "atuação"/corpo/pose de fundo) — o rosto original detectado nesses vídeos
é sempre substituído, nunca reconstruído/imitado.

Essa regra é garantida estruturalmente:
- `datasets/scene_dataset.py` (`SceneDataset` e subclasses) nunca expõe
  identidade/label de quem aparece no vídeo — só `path/dataset_name/split`.
- `datasets/identity_dataset.py` (`IdentityDataset`) só aceita uma pasta local
  do usuário e exige um `consent.yaml` válido (veja
  `configs/example_identity_consent.yaml`) antes de liberar qualquer dado.
- `training/trainer.py` (`Trainer.fit`) exige o manifesto de consentimento
  como argumento obrigatório e valida os tipos dos datasets recebidos —
  tentar usar uma `SceneDataset` como identidade é um erro de tipo, não um
  uso silencioso.

Não use esta ferramenta para gerar conteúdo de pessoas reais sem
consentimento, para fraude, difamação ou desinformação.

## Datasets de pesquisa (FF++ / Celeb-DF / DFDC)

Esta ferramenta **não baixa nem faz scraping** desses datasets — todos exigem
registro/aceite de termos acadêmicos próprios. Rode:

```
python cli.py dataset-info faceforensics
python cli.py dataset-info celebdf
python cli.py dataset-info dfdc
```

para ver o link oficial de registro e o layout de pastas esperado por cada
loader. Depois de baixar por conta própria, aponte `--dataset-root` para a
pasta local.

## Setup (Windows)

O PyTorch, insightface e onnxruntime ainda não têm wheels para Windows no
Python 3.14. É necessário Python 3.10 ou 3.11 dedicado a este projeto:

```powershell
# Depois de instalar o Python 3.10/3.11 (ex: py -3.11 disponível):
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

GPU (CUDA) é detectada automaticamente (`utils/device.py`); sem GPU, o
pipeline roda em CPU (mais lento — recomenda-se resolução 128px e poucas
fotos de identidade).

## Interface gráfica (opcional)

Em vez de linha de comando, há uma GUI desktop (Tkinter) com 3 abas:

```
python gui.py
```

- **1. Identidade**: escolha um nome, adicione suas fotos/vídeo (botões
  "Adicionar fotos.../Adicionar vídeo..."), preencha e salve o
  `consent.yaml` direto pela tela (o status mostra se o consentimento está
  válido).
- **2. Cena**: escolha `custom` (vídeo/pasta próprios) ou um dataset de
  pesquisa + o caminho local já baixado; "Ver instruções de acesso" mostra
  como obter cada dataset; "Testar/contar vídeos" valida o caminho.
- **3. Pipeline**: botões para alinhar identidade/cena, treinar e gerar o
  vídeo final, com log em tempo real na parte de baixo da janela.

A janela em si só precisa de Tkinter (já vem com o Python) + `pyyaml`. A
aba "Cena" precisa de `numpy`/`opencv-python`; a aba "Pipeline" (align
com insightface, treino/inferência com PyTorch) precisa do
`requirements.txt` completo instalado no venv Python 3.10/3.11 — sem isso,
os botões dessa aba mostram um erro explicando o que falta, mas as
abas 1 e 2 continuam utilizáveis.

## Fluxo de uso (linha de comando)

1. `python cli.py dataset-info <nome>` — instruções de acesso ao dataset (ou pule e use um vídeo próprio).
2. Coloque fotos/vídeo da identidade em `data/identities/<nome>/`, copie
   `configs/example_identity_consent.yaml` para
   `data/identities/<nome>/consent.yaml` e preencha (`consent_confirmed: true`).
3. `python cli.py align --identity <nome> --sanity-check` — gera crops
   alinhados e um preview com landmarks para conferência visual.
4. `python cli.py align --scene <video_ou_dataset> --sanity-check` — idem
   para a cena.
5. `python cli.py train --identity <nome> --scene <...> --epochs 200`
6. `python cli.py infer --identity <nome> --scene <...> --output data/outputs/out.mp4`

## Limitações conhecidas (v1)

- Uma identidade por vez.
- Heurística de "maior rosto por frame" — não há rastreamento de pessoa
  específica em cenas com múltiplos rostos.
- Blend padrão é color-transfer LAB + feather alpha; `--blend poisson` usa
  `cv2.seamlessClone` (mais lento, ocasionalmente com artefatos).
