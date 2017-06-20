(ns machine-learning.report.image
  (:require [clojure.java.io :as io])
  (:import (java.awt.image BufferedImage)
           (java.awt Color)
           (javax.imageio ImageIO)
           (java.io File)))

(defn as-rgb [intensity]
  (let [gray (Math/round ^Float (* (- 1 intensity) 255.0))]
    (.getRGB (Color. gray gray gray))))

(defn set-pixel [b_image pos pixel]
  (.setRGB b_image (quot pos 28) (mod pos 28) (as-rgb pixel)))

(defn image-for [image]
  (let [bi (BufferedImage. 28 28 BufferedImage/TYPE_INT_RGB)]
    (dorun (map  #(set-pixel bi %1 (first %2)) (range 0 784) image))
    bi))

(defn save-image-png! [image filename]
  (let [^BufferedImage bi (image-for image)
        g (.createGraphics bi)]
    (ImageIO/write bi "png" ^File (io/as-file filename))))

(defn save-images! [eval test_data dir]
  (let [^File dir (io/file dir)]
    (.mkdirs dir)
    (map #(save-image-png! (first (get test_data %)) (io/file dir (format "%s.png" %))) eval)))
