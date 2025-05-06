```mermaid
flowchart LR
  subgraph Człowiek
    IDEEP([I.DEEP 🧘]) --> IUT([I.UT 🗣️])
  end
  subgraph Sztuczna_Inteligencja
    AIUT([AI.UT 💬]) --> AIDEEP([AI.DEEP ⚙️])
  end
  IUT --> AIUT
  AIDEEP -- feedback --> IDEEP
