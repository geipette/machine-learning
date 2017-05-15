(defproject machine-learning "0.1.0-SNAPSHOT"
  :description "Testing machine learning in clojure"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.clojure/tools.logging "0.3.1"]
                 [ch.qos.logback/logback-classic "1.2.3"]
                 [net.mikera/core.matrix "0.59.0"]
                 [uncomplicate/neanderthal "0.10.0"]]
  :main ^:skip-aot machine-learning.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
