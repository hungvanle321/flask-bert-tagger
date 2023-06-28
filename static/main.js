const data = [
  ["ADP", "Adposition", "#e53935"],
  ["DET", "Determiner", "#6a1b9a"],
  ["PROPN", "Proper Noun", "#0277bd"],
  ["VERB", "Verb", "#d81b60"],
  ["NOUN", "Noun", "#689f38"],
  ["PUNCT", "Punctuation", "#f9a825"],
  ["NUM", "Numeral", "#3e20aa"],
  ["PART", "Particle", "#4e342e"],
  ["ADJ", "Adjective", "#f57f17"],
  ["ADV", "Adverb", "#00897b"],
  ["AUX", "Auxiliary Verb", "#2135b1"],
  ["INTJ", "Interjection", "#1fb342"],
  ["PRON", "Pronoun", "#039be5"],
  ["CCONJ", "Coordinating Conjunction", "#2e7d32"],
  ["SCONJ", "Subordinating Conjunction", "#b36654"],
  ["X", "Other", "#2e2827"],
  ["SYM", "Symbol", "#90840e"],
];


const example = document.querySelector("#example");
if (example) {
  example.textContent = "The result after defining part of speech...";
  example.addEventListener("focus", () => {
    example.textContent = "";
  });
}

const container = document.getElementById("tags-container");
if (container){
  data.forEach((item) => {
    const div = document.createElement("div");
    div.classList.add("tag-textblock");
    div.textContent = item[1];
    div.style.backgroundColor = item[2];
    div.style.color = "#ffffff";
    container.appendChild(div);
  });
}

function decorated(POS_list) {
  POS_list.forEach((i) => console.log(i));

  const container = document.getElementById("example");
  container.innerHTML = "";

  POS_list.forEach((word_pos) => {
    const div = document.createElement("div");
    const data_find = data.find((item) => item[0] === word_pos[1]);
    div.classList.add("result-textblock");
    div.textContent = word_pos[0];
    div.style.backgroundColor = data_find[2];
    div.style.color = "#ffffff";
    div.title = data_find[1];
    container.appendChild(div);
  });
}
