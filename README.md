# Similarity-Attributes
---
Penggunaan Similarity Measure dalam melakukan pencarian atau pencocokan atribut telah banyak digunakan oleh perusahaan dalam administrasi transaksi. Tentunya ada beberapa algoritma perhitungan berdasarkan jenis metodenya antara lain:
* Edit-based
  Merupakan algoritma yang melakukan perhitungan perubahan huruf melalui operasi (Insertion/Deletion/Substitution/Transposition).
  Terdapat beberapa algoritma perhitungan, antara lain:
  1. Jaro Distance
  2. Jaro-Winkler Similarity
  3. Levenshtein Distance
  4. Damerau-Levenshtein
  5. Hamming Distance (Both strings must have the same length)
     
* Token-based
  Merupakan algoritma yang mengubah beberapa kalimat menjadi suku kata dan dilakukan perbandingan dengan kalimat lainnya.
  Terdapat beberapa algoritma perhitungan, antara lain:
  1. Euclidean Distance
  2. Minkowski Distance
  3. Manhattan Distance
  4. Cosine Similarity

Adapun data dummy yang akan digunakan sebagai contoh, berikut struktur data yang digunakan:

!["Data Structure"](https://github.com/pinantyo/Similarity-Attributes/blob/main/assets/Data.png?raw=true)


Bila menggunakan algoritma clustering K-Means, berdasarkan kebutuhan bisnis, perlunya bobot yang diberikan untuk masing-masing atribut. Hal ini membuat perlunya pengubahan algoritma K-Means untuk menerima pembobotan tersebut. Adapun algoritma yang dirancang adalah sebagai berikut:

!["Algorithm Structure"](https://github.com/pinantyo/Similarity-Attributes/blob/main/assets/Model.png?raw=true)
