# Wycena opcji ogonowych metodą prawa potęgowego Taleba

## 1. Źródło i cel metody

Dokument opisuje metodę z artykułu:

> Nassim Nicholas Taleb, Brandon Yarckin, Chitpuneet Mann, Damir Delic i Mark Spitznagel, *Tail Option Pricing Under Power Laws*, rewizja z marca 2023 r., arXiv:1908.02347v3.

Metoda służy przede wszystkim do **relatywnej wyceny opcji położonych w dalekim ogonie**. Nie próbuje wyznaczyć absolutnej wartości opcji od zera. Zamiast tego:

1. wybiera płynną opcję ogonową o znanej cenie rynkowej,
2. traktuje ją jako punkt zakotwiczenia,
3. zakłada potęgowy przebieg dalszego ogona,
4. wyznacza ceny bardziej odległych strike'ów przy użyciu parametru ogona \(\alpha\).

Cena opcji kotwiczącej zawiera już rynkową ocenę zmienności, stóp procentowych, dywidend, premii za ryzyko i bieżącego popytu na zabezpieczenie. Dzięki użyciu relacji cen znika potrzeba osobnego szacowania większości parametrów skali rozkładu.

Metoda nie jest zamiennikiem całej powierzchni zmienności. Jest narzędziem do **ekstrapolacji poza obszar dobrze obserwowanych strike'ów** oraz do oceny, czy relacje cen w ogonie są zgodne z zadanym prawem potęgowym.

---

## 2. Regularna zmienność i prawo potęgowe

Punktem wyjścia jest funkcja przeżycia zmiennej losowej \(X\):

$$
\Pr(X>x) \sim L(x)x^{-\alpha},
$$

gdzie:

- \(\alpha>1\) — indeks ogona,
- \(L(x)\) — funkcja wolnozmienna,
- symbol \(\sim\) oznacza asymptotyczną równoważność.

Funkcja \(L\) jest wolnozmienna, jeżeli dla każdego \(k>0\):

$$
\lim_{x\to\infty}\frac{L(kx)}{L(x)}=1.
$$

Powyżej odpowiednio wysokiego progu autorzy przyjmują, że \(L(x)\) można bezpiecznie zastąpić stałą. Próg ten nazywają **stałą Karamaty**. W tym obszarze ogon zachowuje się jak silne prawo Pareto.

Interpretacja parametru \(\alpha\):

- mniejsza wartość \(\alpha\) oznacza grubszy ogon,
- większa wartość \(\alpha\) oznacza szybszy zanik prawdopodobieństwa,
- \(\alpha>1\) zapewnia istnienie pierwszego momentu,
- skończona wariancja wymaga \(\alpha>2\), ale nie jest konieczna do zastosowania relatywnej wyceny opcji.

---

## 3. Oznaczenia

| Symbol | Znaczenie |
|---|---|
| \(S_0\) | bieżąca cena instrumentu bazowego |
| \(S_T\) | cena instrumentu w terminie wygaśnięcia |
| \(K\) | strike opcji |
| \(K_1\) | strike opcji kotwiczącej |
| \(K_2\) | strike wycenianej opcji ogonowej |
| \(C(K)\) | cena calla o strike'u \(K\) |
| \(P(K)\) | cena puta o strike'u \(K\) |
| \(C_m\), \(P_m\) | obserwowana cena rynkowa opcji kotwiczącej |
| \(l\) | próg wejścia w obszar Pareto, wyrażony jako cena albo stopa zwrotu |
| \(\alpha\) | indeks ogona Pareto |
| \(T\) | czas do wygaśnięcia w latach |

Wszystkie porównywane opcje muszą dotyczyć tego samego instrumentu bazowego i tego samego terminu wygaśnięcia.

---

## 4. Wycena calli w prawym ogonie

### 4.1. Prawo potęgowe bezpośrednio dla ceny \(S_T\)

W uproszczonym podejściu zakłada się, że sama przyszła cena należy w prawym ogonie do klasy regularnie zmiennych rozkładów. Dla \(K>l>0\) i \(\alpha>1\):

$$
C(K)=\frac{K^{1-\alpha}l^\alpha}{\alpha-1}\,\nu,
$$

gdzie \(\nu\) jest stałą normalizującą. Po zakotwiczeniu w rynkowej cenie \(C(K_1)\) parametry \(l\) i \(\nu\) redukują się:

$$
\boxed{
C(K_2)=C(K_1)\left(\frac{K_2}{K_1}\right)^{1-\alpha}
}
$$

dla strike'ów leżących w obszarze Pareto.

### 4.2. Prawo potęgowe dla dodatnich zwrotów

Bardziej naturalnym założeniem jest prawo potęgowe dla geometrycznego zwrotu:

$$
R_+=\frac{S_T-S_0}{S_0}.
$$

Dla \(R_+>l\) funkcja przeżycia ma postać:

$$
\Pr(R_+>r)=\left(\frac{r}{l}\right)^{-\alpha}.
$$

Call dla \(K\ge (1+l)S_0\) ma cenę proporcjonalną do:

$$
C(K,S_0)=\frac{(lS_0)^\alpha(K-S_0)^{1-\alpha}}{\alpha-1}.
$$

Po zakotwiczeniu:

$$
\boxed{
C(K_2)=C(K_1)
\left(\frac{K_2-S_0}{K_1-S_0}\right)^{1-\alpha}
}
$$

dla \(K_1,K_2\ge(1+l)S_0\).

Próg \(l\) wynikający z ceny kotwiczącej \(C_m=C(K_1)\) można zapisać jako:

$$
l=
\frac{\left[(\alpha-1)C_m\right]^{1/\alpha}
(K_1-S_0)^{1-1/\alpha}}{S_0}.
$$

W praktycznej relacji pomiędzy dwiema opcjami próg ten nie musi być jawnie estymowany.

---

## 5. Wycena putów w lewym ogonie — Result 3

Dla lewego ogona definiujemy dodatnią wielkość straty:

$$
R_- = \frac{S_0-S_T}{S_0},
\qquad S_T=(1-R_-)S_0.
$$

Zakładamy, że dla \(R_->l\) strata ma ogon Pareto o indeksie \(\alpha\). Rozpatrywane strike'i spełniają:

$$
K_1,K_2\le(1-l)S_0.
$$

### 5.1. Bezwymiarowy kernel puta

Po scałkowaniu wypłaty \((K-S_T)^+\) względem potęgowej gęstości straty otrzymujemy czynnik zależny od strike'u:

$$
\boxed{
H(K;S_0,\alpha)=
\left(1-\frac{K}{S_0}\right)^{1-\alpha}
-1-(\alpha-1)\frac{K}{S_0}
}
$$

dla \(0<K<S_0\) oraz \(\alpha>1\).

Wzór relatywnej wyceny ma wtedy postać:

$$
\boxed{
P(K_2)=P(K_1)
\frac{H(K_2;S_0,\alpha)}{H(K_1;S_0,\alpha)}
}
$$

To jest postać używana w `analiza_opcji.py` przez funkcje:

- `result3_put_kernel`,
- `result3_relative_put_prices`.

### 5.2. Uwaga dotycząca równania (7) w artykule

W wydrukowanym równaniu (7) występuje zapis z potęgami \((K-S_0)^{1-\alpha}\) oraz \(S_0^{1-\alpha}\). W tej formie wyrażenie jest problematyczne dla niecałkowitego \(\alpha\), ponieważ \(K-S_0<0\), a człony nie są wymiarowo zgodne.

Dlatego implementacja nie kopiuje równania typograficznie. Używa powyższego bezwymiarowego kernela \(H(K)\), wynikającego bezpośrednio z całkowania gęstości podanej w sekcji o putach. Zachowuje on sens Result 3, daje wartości rzeczywiste dla dowolnego \(\alpha>1\) i jest wymiarowo spójny.

### 5.3. Zachowanie asymptotyczne

W dalekim ogonie dominująca relacja ma postać:

$$
P(K)\propto(S_0-K)^{1-\alpha}.
$$

Po podzieleniu przez cenę opcji kotwiczącej:

$$
\frac{P(K)}{P(K_1)}
\approx
\left(\frac{S_0-K}{S_0-K_1}\right)^{1-\alpha}.
$$

Po zalogarytmowaniu:

$$
\log\frac{P(K)}{P(K_1)}
\approx
(1-\alpha)
\log\frac{S_0-K}{S_0-K_1}.
$$

Na wykresie log–log otrzymujemy zatem w przybliżeniu prostą o nachyleniu:

$$
\beta=1-\alpha,
\qquad
\boxed{\alpha=1-\beta}.
$$

Dla \(\alpha=3\) teoretyczne nachylenie wynosi \(-2\). Jeżeli rynek ma nachylenie mniej ujemne, na przykład \(-1{,}8\), odpowiada to \(\alpha\approx2{,}8\), czyli grubszemu ogonowi niż przy \(\alpha=3\).

---

## 6. Dobór opcji kotwiczącej

Kotwica jest najważniejszym rynkowym wejściem do metody. Powinna:

- znajdować się już w obszarze, w którym założenie Pareto jest wiarygodne,
- mieć możliwie duży wolumen i open interest,
- mieć wąski spread bid–ask,
- posiadać świeżą cenę,
- należeć do tego samego terminu wygaśnięcia co pozostałe opcje,
- nie być tak odległa, aby pojedyncze niereprezentatywne transakcje ustalały jej cenę.

Do obliczeń najlepiej używać jednoczesnych kwotowań. W zależności od celu można stosować:

- midpoint \((bid+ask)/2\) — do opisu powierzchni,
- ask — gdy analizujemy koszt zakupu zabezpieczenia,
- bid — gdy analizujemy realną cenę sprzedaży,
- settlement — gdy potrzebna jest oficjalna cena rozliczeniowa.

`lastPrice` może pochodzić ze starej transakcji i nie musi reprezentować aktualnego rynku.

---

## 7. Przeliczenie strike'u na liczbę odchyleń standardowych

Metoda prawa potęgowego sama w sobie nie potrzebuje zmienności. Przeliczenie na \(\sigma\) jest jednak wygodne do opisu położenia kotwicy.

Z ATM implied volatility:

$$
\sigma_T=\sigma_{ATM}\sqrt{T}.
$$

Proste przybliżenie procentowe:

$$
z_{simple}=\frac{K/S_0-1}{\sigma_{ATM}\sqrt{T}}.
$$

W modelu lognormalnym BSM dokładniejsza miara wynosi:

$$
z_{BSM}(K)=
\frac{
\ln(K/S_0)-
\left(r-q-\tfrac12\sigma_{ATM}^2\right)T
}{\sigma_{ATM}\sqrt{T}},
$$

gdzie \(r\) jest stopą wolną od ryzyka, a \(q\) stopą dywidendy lub convenience yield.

Liczba \(\sigma\) jest tu tylko współrzędną opartą na ATM IV. Nie oznacza, że faktyczny rozkład jest normalny. Przy silnym skew IV dalekie strike'i mogą mieć znacznie wyższą zmienność implikowaną niż ATM.

### Przykład z analizowanego łańcucha

Dla danych SPX z 16 października 2026 r., użytych jako proxy dla ES:

- \(S_0=7659{,}44\),
- termin: 30 dni,
- \(\sigma_{ATM}=12{,}89\%\),
- \(K_1=6900\),
- \(P(K_1)=21{,}60\).

Otrzymujemy:

$$
\sigma_T=12{,}89\%\sqrt{30/365}=3{,}70\%.
$$

Strike 6900 leży około:

- \(-2{,}68\sigma\) w prostym przybliżeniu procentowym,
- \(-2{,}88\sigma\) według lognormalnej miary BSM uwzględniającej forward.

Praktycznie kotwicę można więc opisać jako około **\(-2{,}9\sigma\) ATM dla 30 DTE**.

---

## 8. Relacja do BSM i Breedena–Litzenbergera

### Black–Scholes–Merton

BSM zakłada lognormalny rozkład ceny końcowej przy stałej zmienności. Daje kompletną cenę opcji po podaniu \(S_0,K,T,r,q,\sigma\). W praktyce różne strike'i mają różne IV, dlatego obserwujemy smile lub skew.

Metoda Taleba nie dopasowuje zmienności dla każdego strike'u. Zakłada potęgowy kształt dalekiego ogona i wycenia opcje **relatywnie do ceny kotwiczącej**.

### Breeden–Litzenberger

Dla europejskich opcji i odpowiednio gładkiej krzywej cen:

$$
\frac{\partial^2 C(K,T)}{\partial K^2}
=e^{-rT}f_Q(K),
$$

oraz analogicznie:

$$
\frac{\partial^2 P(K,T)}{\partial K^2}
=e^{-rT}f_Q(K),
$$

gdzie \(f_Q\) jest gęstością ceny końcowej w mierze neutralnej względem ryzyka.

Breeden–Litzenberger **odtwarza lokalną gęstość z obserwowanej krzywej cen opcji**. Metoda Taleba **ekstrapoluje relacje cen dalej w ogon** według prawa potęgowego. Podejścia mogą się uzupełniać:

1. rynek i BL określają obserwowany obszar oraz możliwą kotwicę,
2. prawo potęgowe opisuje dalszą część ogona,
3. porównanie z BSM pokazuje, jak bardzo ogon odbiega od lognormalności.

---

## 9. Jak interpretować cenę rynkową wyższą od krzywej Pareto

Jeżeli:

$$
P_{market}(K)>P_{Pareto,\alpha}(K),
$$

to opcja zawiera premię względem **konkretnej** krzywej o zadanym \(\alpha\) i wybranej kotwicy. Może to oznaczać:

- rynek implikuje mniejsze \(\alpha\), czyli grubszy ogon,
- wzrósł popyt na ochronę,
- sprzedający żądają premii za ryzyko skoku i ryzyko bilansowe,
- kwotowanie ma szeroki spread albo niską płynność,
- `lastPrice` jest nieaktualna,
- wybrana kotwica nie znajduje się jeszcze w stabilnym obszarze Pareto.

Nie jest to automatyczny dowód, że opcja jest ekonomicznie przewartościowana. Cena opcji opisuje rozkład neutralny względem ryzyka i premię za ubezpieczenie, a nie wyłącznie fizyczne prawdopodobieństwo spadku.

Autorzy metody podkreślają, że jej podstawowym przedmiotem jest **względna wycena względem bliższego strike'u**, a nie dowodzenie absolutnego mispricingu całego ogona.

---

## 10. Kontrola arbitrażu i jakości danych

Przed użyciem wyniku należy sprawdzić co najmniej:

1. **Monotoniczność:** cena puta powinna rosnąć wraz ze strike'em.
2. **Wypukłość:** krzywa ceny względem strike'u powinna być wypukła.
3. **Granice cenowe:** dla europejskiego puta

   $$
   \max\left(Ke^{-rT}-S_0e^{-qT},0\right)
   \le P(K,T)\le Ke^{-rT}.
   $$

4. **Spójność czasu:** kwotowania muszą pochodzić z możliwie tego samego momentu.
5. **Spójność kontraktów:** ten sam styl wykonania, mnożnik, sposób rozliczenia i termin.
6. **Płynność:** spread, wolumen, open interest i wiek transakcji.
7. **Dywidendy i forward:** szczególnie gdy zamiast indeksu używany jest futures albo proxy.

---

## 11. Praktyczny algorytm

1. Wybierz instrument oraz jeden termin wygaśnięcia.
2. Pobierz spot lub właściwy forward i pełny łańcuch opcji.
3. Odfiltruj nieświeże oraz nielikwidne kwotowania.
4. Wyznacz obszar, w którym ogon może zachowywać się potęgowo.
5. Wybierz płynną kotwicę \((K_1,P(K_1))\).
6. Przyjmij \(\alpha\) z badań historycznych albo oszacuj je z przekroju cen.
7. Dla każdego dalszego puta oblicz:

   $$
   P(K)=P(K_1)\frac{H(K)}{H(K_1)}.
   $$

8. Porównaj wynik z bidem, askiem, midpointem, BSM i gęstością BL.
9. Sprawdź ograniczenia arbitrażowe.
10. Powtórz analizę dla innych terminów DTE; nie przenoś automatycznie jednej kotwicy pomiędzy terminami.

---

## 12. Ograniczenia

- Wynik silnie zależy od jakości i wyboru ceny kotwiczącej.
- \(\alpha\) nie musi być stałe pomiędzy terminami ani reżimami rynkowymi.
- Nie wiadomo z góry, od którego strike'u zaczyna się stabilny obszar Karamaty.
- Dalekie opcje często mają szerokie spready i rzadkie transakcje.
- Metoda opisuje przekrój strike'ów dla jednego terminu, a nie pełną dynamikę procesu cenowego.
- Użycie indeksu SPX jako proxy dla opcji na futures ES wprowadza różnice w instrumencie bazowym, godzinach handlu, rozliczeniu i mikrostrukturze.
- Prawo potęgowe nie usuwa potrzeby kontroli statycznego arbitrażu.

---

## 13. Najważniejszy sens ekonomiczny

BSM pyta: **ile kosztowałaby opcja przy lognormalnym rozkładzie i zadanej zmienności?**

Breeden–Litzenberger pyta: **jaką gęstość neutralną względem ryzyka implikują obserwowane ceny opcji?**

Metoda Taleba pyta: **jeśli rynek podał cenę jednej opcji w ogonie, jakie powinny być relatywne ceny jeszcze dalszych opcji, gdy ogon podlega prawu potęgowemu o indeksie \(\alpha\)?**

To ostatnie pytanie jest szczególnie użyteczne tam, gdzie BSM szybko wygasza ceny, a dane rynkowe stają się rzadkie i niestabilne.

---

## 14. Materiały powiązane

- Artykuł: [Tail Option Pricing Under Power Laws](https://arxiv.org/abs/1908.02347)
- Implementacja: `analiza_opcji.py`
- Wyniki i wykresy: katalog `bsm_vs_market/`

Dokument ma charakter analityczny i edukacyjny. Nie stanowi rekomendacji inwestycyjnej ani kompletnego modelu zarządzania ryzykiem.
