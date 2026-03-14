;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "predicting_drosophila_body_orientation_from_a_translational_trajectory_using_an_artificial_neural_network"
 (lambda ()
   (setq TeX-command-extra-options
         "-shell-escape")
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "11pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art11"
    "inputenc"
    "fontenc"
    "amsmath"
    "amssymb"
    "graphicx"
    "hyperref"
    "bm"
    "natbib")
   (LaTeX-add-labels
    "eq:heading_correction_opt")
   (LaTeX-add-bibliographies
    "main"))
 :latex)

