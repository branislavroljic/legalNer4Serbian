# Complete manual annotations for few-shot examples - ALL 14 entity types
# Based on actual content analysis of the few-shot example files

MANUAL_ANNOTATIONS_COMPLETE = {
    "example_01_theft.txt": {
        "sud ili tribunal": ["Osnovni sud u Herceg Novom"],
        "datum presude ili odluke": ["30.12.2019."],
        "broj predmeta ili identifikator slučaja": ["K. br. 245/23"],
        "krivično delo ili prestup": ["krađe"],
        "tužilac ili javni tužilac": ["Dragana Milovanovića"],
        "optuženi ili osoba na suđenju": ["M.P."],
        "sudija ili pravosudni službenik": ["Marija Nikolić"],
        "sudski zapisničar ili službenik": ["Ane Stojanović"],
        "sudska presuda ili odluka": ["USLOVNU OSUDU"],
        "vrsta kazne ili sankcije": ["kaznu zatvora"],
        "iznos ili trajanje kazne": ["6 mjeseci"],
        "materijalna pravna odredba ili član": [
            "člana 344. stav 1. Krivičnog zakonika"
        ],
        "procesna pravna odredba ili član": [
            "članu 434. Zakonika o krivičnom postupku",
            "člana 261. Zakonika o krivičnom postupku",
        ],
        "troskovi ili takse sudskog postupka": ["40€"],
    },
    "example_02_assault.txt": {
        "sud ili tribunal": ["Viši sud u Podgorici"],
        "datum presude ili odluke": ["16.05.2019."],
        "broj predmeta ili identifikator slučaja": ["Kž. 1567/22"],
        "krivično delo ili prestup": ["nasilničko ponašanje"],
        "tužilac ili javni tužilac": ["Milana Đorđevića"],
        "optuženi ili osoba na suđenju": ["S.M."],
        "sudija ili pravosudni službenik": [
            "Aleksandar Jovanović",
            "Milica Radović",
            "Petar Stanković",
        ],
        "sudski zapisničar ili službenik": ["Jovane Mitrović"],
        "sudska presuda ili odluka": ["OSLOBAĐA SE OD OPTUŽBE"],
        "vrsta kazne ili sankcije": [],
        "iznos ili trajanje kazne": [],
        "materijalna pravna odredba ili član": [
            "člana 220. stav 1. Krivičnog zakonika"
        ],
        "procesna pravna odredba ili član": [
            "članu 434. Zakonika o krivičnom postupku"
        ],
        "troskovi ili takse sudskog postupka": [],
    },
    "example_03_fraud.txt": {
        "sud ili tribunal": ["Osnovni sud u Nikšiću", "OSNOVNI SUD U NIKŠIĆU"],
        "datum presude ili odluke": ["28. juna 2023. godine"],
        "broj predmeta ili identifikator slučaja": ["K. br. 567/23"],
        "krivično delo ili prestup": ["prevare"],
        "tužilac ili javni tužilac": ["Srđana Petrovića"],
        "optuženi ili osoba na suđenju": ["A.S."],
        "sudija ili pravosudni službenik": ["Jelena Milosavljević"],
        "sudski zapisničar ili službenik": [
            "Nemanje Stojanovića",
            "Nemanja Stojanović",
        ],
        "sudska presuda ili odluka": ["O S U Đ U J E"],
        "vrsta kazne ili sankcije": ["kaznu zatvora", "uslovnu osudu"],
        "iznos ili trajanje kazne": ["8 mjeseci", "sa rokom kušnje od 2 godine"],
        "materijalna pravna odredba ili član": [
            "člana 208. stav 1. Krivičnog zakonika"
        ],
        "procesna pravna odredba ili član": [],
        "troskovi ili takse sudskog postupka": ["120€"],
    },
    "example_04_traffic.txt": {
        "sud ili tribunal": [
            "PREKRŠAJNI SUD U PODGORICI",
            "Prekršajni sud u Podgorici",
        ],
        "datum presude ili odluke": ["10. maja 2023. godine"],
        "broj predmeta ili identifikator slučaja": ["Pr. br. 3456/23"],
        "krivično delo ili prestup": ["prekršaj"],
        "tužilac ili javni tužilac": [],
        "optuženi ili osoba na suđenju": ["V.M."],
        "sudija ili pravosudni službenik": ["Marko Đorđević"],
        "sudski zapisničar ili službenik": ["Tanje Nikolić"],
        "sudska presuda ili odluka": ["P R E S U D U"],
        "vrsta kazne ili sankcije": [
            "novčanom kaznom",
            "ZAŠTITNA MERA zabrane upravljanja motornim vozilom",
        ],
        "iznos ili trajanje kazne": ["80€", "u trajanju od 8 mjeseci"],
        "materijalna pravna odredba ili član": [
            "člana 330. stav 1. tačka 3. Zakona o bezbjednosti saobraćaja na putevima"
        ],
        "procesna pravna odredba ili član": ["članu 175. Zakona o prekršajima"],
        "troskovi ili takse sudskog postupka": ["30€"],
    },
    "example_05_drug_possession.txt": {
        "sud ili tribunal": ["Osnovni sud u Baru"],
        "datum presude ili odluke": ["20.10.2023."],
        "broj predmeta ili identifikator slučaja": ["K. br. 567/23"],
        "krivično delo ili prestup": [
            "neovlašćena proizvodnja i stavljanje u promet opojnih droga"
        ],
        "tužilac ili javni tužilac": [
            "Marije Stanković"
        ],
        "optuženi ili osoba na suđenju": ["N.R."],
        "sudija ili pravosudni službenik": ["Ana Popović"],
        "sudski zapisničar ili službenik": ["Milan Jovanović"],
        "sudska presuda ili odluka": ["USLOVNU OSUDU"],
        "vrsta kazne ili sankcije": ["kaznu zatvora", "Oduzima se predmet"],
        "iznos ili trajanje kazne": ["6 mjeseci", "marihuana"],
        "materijalna pravna odredba ili član": [
            "člana 246a stav 1. Krivičnog zakonika"
        ],
        "procesna pravna odredba ili član": [
            "članu 423. Zakonika o krivičnom postupku"
        ],
        "troskovi ili takse sudskog postupka": ["80€"],
    },
    "example_06_domestic_violence.txt": {
        "sud ili tribunal": ["OSNOVNI SUD U PLJEVLJIMA", "Osnovni sud u Pljevljima"],
        "datum presude ili odluke": ["25. 01. 2017."],
        "broj predmeta ili identifikator slučaja": ["K. br. 789/23"],
        "krivično delo ili prestup": ["nasilje u porodici"],
        "tužilac ili javni tužilac": ["Dragice Nikolić"],
        "optuženi ili osoba na suđenju": ["Z.M."],
        "sudija ili pravosudni službenik": ["Gordana Milić"],
        "sudski zapisničar ili službenik": ["Jovana Petrovića", "Jovan Petrović"],
        "sudska presuda ili odluka": ["O S U Đ U J E"],
        "vrsta kazne ili sankcije": ["kaznu zatvora", "ZAŠTITNA MERA zabrane prilaska"],
        "iznos ili trajanje kazne": ["10 mjeseci", "u trajanju od 1 godine"],
        "materijalna pravna odredba ili član": [
            "člana 194. stav 2. Krivičnog zakonika CG"
        ],
        "procesna pravna odredba ili član": [],
        "troskovi ili takse sudskog postupka": ["150€"],
    },
    "example_07_embezzlement.txt": {
        "sud ili tribunal": ["VIŠI SUD U PODGORICI", "Viši sud u Podgorici"],
        "datum presude ili odluke": ["09.12.2020"],
        "broj predmeta ili identifikator slučaja": ["K. br. 234/22"],
        "krivično delo ili prestup": ["pronevjere"],
        "tužilac ili javni tužilac": ["Aleksandra Milovanovića"],
        "optuženi ili osoba na suđenju": ["M.P."],
        "sudija ili pravosudni službenik": [
            "Milan Stojanović",
        ],
        "sudski zapisničar ili službenik": ["Milice Jovanović", "Milica Jovanović"],
        "sudska presuda ili odluka": ["O S U Đ U J E"],
        "vrsta kazne ili sankcije": ["kaznu zatvora"],
        "iznos ili trajanje kazne": ["3 godine"],
        "materijalna pravna odredba ili član": [
            "člana 364. stav 3. Krivičnog zakonika"
        ],
        "procesna pravna odredba ili član": ["čl. 363 st. 1 tač. 3 ZKP-a"],
        "troskovi ili takse sudskog postupka": ["450€"],
    },
    "example_08_tax_evasion.txt": {
        "sud ili tribunal": ["OSNOVNI SUD U ROŽAJAMA", "Osnovni sud u Rožajama"],
        "datum presude ili odluke": ["25. 08. 2023."],
        "broj predmeta ili identifikator slučaja": ["K. br. 445/23"],
        "krivično delo ili prestup": ["poreska utaja"],
        "tužilac ili javni tužilac": ["Jovane Petrović"],
        "optuženi ili osoba na suđenju": ["R.Đ."],
        "sudija ili pravosudni službenik": ["Milena Stanković"],
        "sudski zapisničar ili službenik": ["Nemanje Milovanovića", "Nemanja Milovanović"],
        "sudska presuda ili odluka": ["USLOVNU OSUDU"],
        "vrsta kazne ili sankcije": ["kaznu zatvora"],
        "iznos ili trajanje kazne": ["1 godine i 6 mjeseci", "u roku od 3 godine"],
        "materijalna pravna odredba ili član": [
            "člana 229. stav 2. Krivičnog zakonika",
            "čl. 4 st. 2, čl. 5, čl. 13, čl.15, čl. 42 st. 1, čl. 52 st. 2, čl. 53 i čl. 54 Krivičnog zakonika Crne Gore"
        ],
        "procesna pravna odredba ili član": ["čl. 226, čl. 229 i čl. 374 Zakonika o krivičnom postupku"],
        "troskovi ili takse sudskog postupka": ["250€"],
    },
    "example_09_robbery.txt": {
        "sud ili tribunal": ["OSNOVNI SUD U BIJELO POLJU", "Osnovni sud u Bijelo Polju"],
        "datum presude ili odluke": ["06.09.2011."],
        "broj predmeta ili identifikator slučaja": ["K. br. 678/23"],
        "krivično delo ili prestup": ["razbojništva"],
        "tužilac ili javni tužilac": ["Milice Đorđević"],
        "optuženi ili osoba na suđenju": ["M.J.", "S.N."],
        "sudija ili pravosudni službenik": ["Bojan Marković"],
        "sudski zapisničar ili službenik": ["Ane Milenković", "Ana Milenković"],
        "sudska presuda ili odluka": ["O S U Đ U J E"],
        "vrsta kazne ili sankcije": ["kaznu zatvora"],
        "iznos ili trajanje kazne": ["2 godine", "1 godine i 8 mjeseci"],
        "materijalna pravna odredba ili član": [
            "člana 206. stav 1. Krivičnog zakonika",
            "čl.3, 4, 5, 13, 16, 32, 42, 45, 46, 52, 53 i 54 KZ CG"
        ],
        "procesna pravna odredba ili član": ["čl. 226 st.3 i čl.374 ZKP-a"],
        "troskovi ili takse sudskog postupka": ["300€"],
    },
    "example_10_corruption.txt": {
        "sud ili tribunal": ["OSNOVNI SUD U BERANAMA", "Osnovni sud u Beranama"],
        "datum presude ili odluke": ["01.04.2024."],
        "broj predmeta ili identifikator slučaja": ["K. br. 123/22"],
        "krivično delo ili prestup": ["primanja mita"],
        "tužilac ili javni tužilac": ["Nikole Samardžića"],
        "optuženi ili osoba na suđenju": ["V.S."],
        "sudija ili pravosudni službenik": [
            "Vesna Gazdić"
        ],
        "sudski zapisničar ili službenik": ["Ristić Katarina", "Katarine Ristić"],
        "sudska presuda ili odluka": ["O S U Đ U J E"],
        "vrsta kazne ili sankcije": [
            "kaznu zatvora",
            "SPOREDNA KAZNA zabrane vršenja javne funkcije",
        ],
        "iznos ili trajanje kazne": ["4 godine", "u trajanju od 3 godine"],
        "materijalna pravna odredba ili član": [
            "čl.327 st.4 u vezi st.1 Krivičnog Zakonika CG",
            "čl. 2, čl. 3, čl. 4 st. 2, čl. 5, čl. 13 st. 1, čl. 15, čl. 32, čl. 36, čl. 42, čl. 51 st. 1 Krivičnog zakonika Crne Gore"
        ],
        "procesna pravna odredba ili član": ["čl. 226, 229 i 374 Zakonika o krivičnom postupku"],
        "troskovi ili takse sudskog postupka": ["600€"],
    },
}

print(
    f"📚 Complete manual annotations loaded for {len(MANUAL_ANNOTATIONS_COMPLETE)} examples"
)
print(f"🏷️ Each example now includes all 14 entity types where applicable")
print(f"✅ Coverage: All entity types are now represented across the 10 examples")
