

DATASET = [

    # =========================
    # EigenGAN
    # =========================
    {
        "paper": "EigenGAN",
        "questions": [
            "What is the main contribution of EigenGAN?",
            "How does SVD improve GAN training?",
            "What problems of GANs does EigenGAN address?"
        ],
        "ground_truth": {
            "What is the main contribution of EigenGAN?": [
                "SVD", "subspace", "generator", "stability"
            ],
            "How does SVD improve GAN training?": [
                "singular values", "feature space", "stability"
            ],
            "What problems of GANs does EigenGAN address?": [
                "instability", "mode collapse", "training"
            ]
        }
    },

    # =========================
    # MFGAN
    # =========================
    {
        "paper": "MFGAN",
        "questions": [
            "What is MFGAN?",
            "What is the role of multi-kernel filtering?",
            "Why is MFGAN considered a generic generator?"
        ],
        "ground_truth": {
            "What is MFGAN?": [
                "multi-kernel", "generator", "feature extraction"
            ],
            "What is the role of multi-kernel filtering?": [
                "multiple scales", "features", "frequencies"
            ],
            "Why is MFGAN considered a generic generator?": [
                "general", "multiple applications", "architecture"
            ]
        }
    },

    # =========================
    # CMGAN (multi-generator)
    # =========================
    {
        "paper": "CMGAN",
        "questions": [
            "What is the key idea of CMGAN?",
            "How does competition improve GAN performance?"
        ],
        "ground_truth": {
            "What is the key idea of CMGAN?": [
                "multiple generators", "competition"
            ],
            "How does competition improve GAN performance?": [
                "motivation", "performance", "better generation"
            ]
        }
    },

    # =========================
    # Ontology Depth Estimation
    # =========================
    {
        "paper": "Ontology Depth",
        "questions": [
            "What is the role of ontology in this method?",
            "What are monocular cues?"
        ],
        "ground_truth": {
            "What is the role of ontology in this method?": [
                "knowledge", "reasoning", "semantic"
            ],
            "What are monocular cues?": [
                "depth", "single image", "visual cues"
            ]
        }
    }
]