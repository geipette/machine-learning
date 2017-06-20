(ns machine-learning.report.html
  (:require [hiccup.core :as h]
            [clojure.java.io :as io]
            [machine-learning.network :refer :all]
            [clojure.string :as str])
  (:import (java.io File)))

(defrecord ReportSettings [dir])

(defn image-ref [pos]
  [:img {:src (format "images/%s.png" pos)}])

(defn filter-output [output limit]
  (filter (fn [[pos value]] (> value limit)) (map #(vector %1 (first %2)) (range 0 10) output)))

(defn layout-output [output]
  [:table
   (map (fn [[pos value]] [:tr [:td pos] [:td (format "%2.2f%%" (* 100 value))]]) (filter-output output 0.1))])

(defn images [images]
  [:div {:class "datagrid"}
   [:table
    [:thead [:tr [:th "#"] [:th "img"] [:th "output"] [:th "expected"] [:th "details"]]]
    [:tbody (map #(let [{:keys [pos output expected softmax]} %]
                    [:tr [:td pos] [:td (image-ref pos)] [:td output] [:td expected] [:td (layout-output softmax)]])
                 images)]]])


(defn page-html [data]
  (h/html [:html [:head [:link {:rel "stylesheet" :type "text/css" :href "layout.css"}]] [:body (images data)]]))

(defn write-index-html! [report_settings data]
  (let [^File dir (io/file (:dir report_settings))]
    (if-not (.exists dir)
      (.mkdirs dir))
    (if-not (.isDirectory dir)
      (throw (IllegalArgumentException. (format "%s is not a directory" (.toString dir)))))
    (let [html_file (File. dir "index.html")]
      (spit html_file (page-html data)))
    (let [css_file (File. dir "layout.css")]
      (spit css_file (slurp (io/resource "layout.css"))))))

(defn report [network test_data]
  (map (fn [pos] (let [sample (get test_data pos)
                       sigmoids (feed-forward network (first sample))]
                   {:pos      pos
                    :output   (run-test network (first sample))
                    :expected (second sample)
                    :softmax  (softmax sigmoids)}))
       (evaluate-detailed network test_data)))