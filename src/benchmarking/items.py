from .types import CitationExtractionResult, TestItem

TEST_ITEMS: list[TestItem] = [
    (
        "In Gonzalez v. City of Anaheim, 747 F.3d 789 (9th Cir. 2014), the court ruled...",
        CitationExtractionResult(
            cases={"Gonzalez v. City of Anaheim, 747 F.3d 789 (9th Cir. 2014)": 1},
            statutes={},
        ),
    ),
    (
        "The statute 18 U.S.C. § 1961(1) defines racketeering activities...",
        CitationExtractionResult(cases={}, statutes={"18 U.S.C. § 1961(1)": 1}),
    ),
    (
        "Several months later, on February 22, 2018, the Special Counsel charged defendant with, and a grand jury indicted defendant on (i) five counts of subscribing to false income tax returns, in violation of 26 U.S.C. § 7206(1) ",
        CitationExtractionResult(
            cases={}, statutes={"26 U.S.C. § 7206(1)": 1}
        ),  # https://casetext.com/case/united-states-v-manafort-4
    ),
    (
        """ See Niz-Chavez v. Garland, 141 S. Ct. 1474, 1478, 1486 (2021) (statute requires Notice to Appear to be “a single document containing all the information an individual needs to know
about his removal [proceeding]”).
In addition, if the time or place of the hearing is altered after the issuance of a Notice to Appear, DHS or
EOIR can provide the requisite information in a later
document (hereinafter, a “Notice of Change”). See 8 U.S.C. § 1229(a)(2). """,
        CitationExtractionResult(
            cases={"Niz-Chavez v. Garland, 141 S. Ct. 1474, 1478, 1486 (2021)": 1},
            statutes={"8 U.S.C. § 1229(a)(2)": 1},
        ),  # https://www.supremecourt.gov/DocketPDF/22/22-674/286050/20231101115411077_22-674%20and%2022-884%20bsacNILA.pdf
    ),
    (
        """When Congress passed IIRIRA in 1996, it fundamentally reshaped immigration law by heightening the
        consequences of removal proceedings. For example, the new law expanded the list of aggravated felonies that inter alia render a noncitizen removable and disqualified
        from many forms of relief. See INS v. St. Cyr, 533 U.S. 289, 296-297 & n.4 (2001). It imposed substantial bars on
        reentry for many noncitizens who reside in the country unlawfully prior to their removal. See Berrum-Garcia v. Comfort, 390 F.3d 1158, 1167 (10th Cir. 2004). It vastly expanded the agency’s authority to reinstate removal orders, allowing summary removal of noncitizens who have reentered unlawfully after they were removed for
        any reason. See Arevalo v. Ashcroft, 344 F.3d 1, 5 (1st Cir. 2003). And it precluded noncitizens subject to reinstatement from being eligible for most forms of relief. See Fernandez-Vargas v. Gonzales, 548 U.S. 30, 35 (2006). As IIRIRA made the consequences of removal more severe, it also required the government to provide clearer notice when a noncitizen could be subject to removal. Among “the change[s] IIRIRA wrought,” Vartelas v. Holder, 566 U.S. 257, 272 (2012), was the method
        by which noncitizens must be notified about pending removal proceedings.""",
        CitationExtractionResult(
            cases={
                "INS v. St. Cyr, 533 U.S. 289, 296-297 & n.4 (2001)": 1,
                "Berrum-Garcia v. Comfort, 390 F.3d 1158, 1167 (10th Cir. 2004)": 1,
                "Arevalo v. Ashcroft, 344 F.3d 1, 5 (1st Cir. 2003)": 1,
                "Fernandez-Vargas v. Gonzales, 548 U.S. 30, 35 (2006)": 1,
                "Vartelas v. Holder, 566 U.S. 257, 272 (2012)": 1,
            },
            statutes={},
        ),  # https://www.supremecourt.gov/DocketPDF/22/22-674/286050/20231101115411077_22-674%20and%2022-884%20bsacNILA.pdf
    ),
    (
        """Seventh Circuit: The Seventh Circuit also squarely holds that civil RICO excludes “personal injuries and the
    pecuniary losses incurred therefrom.” Doe v. Roe, 958 F.2d 763, 767 (7th Cir. 1992). Horn claims the Seventh
    Circuit later limited its no-personal-injury-recoveries rule to “potential future employment” with a “carveout” for
    “promised or contracted for wages.” BIO 10-11 (citing Evans v. City of Chicago, 434 F.3d 916 (7th Cir. 2006)).
    """,
        CitationExtractionResult(
            cases={
                "Doe v. Roe, 958 F.2d 763, 767 (7th Cir. 1992)": 1,
                "Evans v. City of Chicago, 434 F.3d 916 (7th Cir. 2006)": 1,
            },
            statutes={},
        ),
    ),  # https://www.supremecourt.gov/DocketPDF/23/23-365/306628/20240403141227348_Horn%20Cert%20Reply.pdf
    (
        "To establish a claim under the New York State Human Rights Law, the plaintiff must demonstrate that she suffered discrimination due to her membership in a protected class. See Forrest v. Jewish Guild for the Blind, 3 N.Y.3d 295, 305 (N.Y. 2004). Defendant ABC Corp. argues that the termination was due to documented performance issues, not discrimination. See Ferrante v. Am. Lung Ass’n, 90 N.Y.2d 623, 630 (N.Y. 1997). Plaintiff counters that the reasons provided by ABC Corp. are merely a pretext for age discrimination. See McDonnell Douglas Corp. v. Green, 411 U.S. 792, 804 (1973).",
        CitationExtractionResult(
            cases={
                "Forrest v. Jewish Guild for the Blind, 3 N.Y.3d 295, 305 (2004)": 1,
                "Ferrante v. Am. Lung Ass’n, 90 N.Y.2d 623, 630 (1997)": 1,
                "McDonnell Douglas Corp. v. Green, 411 U.S. 792, 804 (1973)": 1,
            },
            statutes={},
        ),
    ),
    (
        """Other circuits have similarly declined to enforce provisions of this nature. See, e.g. , In re Grand Jury Subpoena (Miller) , 438 F.3d 1141, 1152–53 (D.C. Cir.) (press subpoena regulations and related USAM provision), cert. denied , 545 U.S. 1150, 125 S.Ct. 2977, 162 L.Ed.2d 906 (2005) ; United States v. Wilson , 413 F.3d 382, 389 (3d Cir. 2005) (Petite policy limiting dual prosecution); United States v. Lee , 274 F.3d 485, 492–93 (8th Cir.) (death penalty protocol), cert. denied , 537 U.S. 1000, 123 S.Ct. 513, 154 L.Ed.2d 394 (2002). Here is a fake citation added to this otherwise real text. Wilson, 413 F.3d at 391.
    """,
        CitationExtractionResult(
            cases={
                "In re Grand Jury Subpoena (Miller), 438 F.3d 1141, 1152–53 (D.C. Cir.), cert. denied, 545 U.S. 1150, 125 S.Ct. 2977, 162 L.Ed.2d 906 (2005)": 1,
                "United States v. Wilson, 413 F.3d 382, 389 (3d Cir. 2005)": 2,
                "United States v. Lee, 274 F.3d 485, 492–93 (8th Cir.), cert. denied, 537 U.S. 1000, 123 S.Ct. 513, 154 L.Ed.2d 394 (2002)": 1,
            },
            statutes={},
        ),
    ),  # https://casetext.com/case/united-states-v-manafort-4
    (
        """On September 7, 2018, the Court sentenced Defendant George Papadopoulos to 14 days of incarceration, one year of supervised release, 200 hours of community service, and a fine of $9,500 for making false statements to the Federal Bureau of Investigation ("FBI") in violation of 18 U.S.C. § 1001. The time to appeal his conviction or sentence expired on September 25, 2018, and he is scheduled to surrender to the Bureau of Prisons to begin serving his term of incarceration on Monday, November 26, 2018.""",
        CitationExtractionResult(cases={}, statutes={"18 U.S.C. § 1001": 1}),
    ),  # https://casetext.com/case/united-states-v-papadopoulos
    (
        """Third, Papadopoulos "claimed he met a certain female Russian national before he joined the Campaign" and that "their communications consisted of emails such as, 'Hi, how are you?'" Id. at 2-3 (SOF ¶ 2c). """,
        CitationExtractionResult(cases={}, statutes={}),
    ),  # https://casetext.com/case/united-states-v-papadopoulos
    (
        """In addition, he waived "any right to challenge [his] conviction . . . or sentence . . . or otherwise attempt to modify or change the  sentence or the manner in which it was determined in any collateral attack, including, but not limited to, a motion brought under 28 U.S.C. § 2255 or Federal Rule of Civil Procedure 60(b), except to the extent such a motion is based on newly discovered evidence or on a claim that [he] received ineffective assistance of counsel."...the Court sentenced Papadopoulos to a within-guidelines sentence of 14 days of incarceration, one year of supervised release, 200 hours of community service, and a fine of $9,500. The Court allowed Papadopoulos to self-surrender at an institution designated by the Bureau of Prisons, as notified by the Probation Office. Under these terms, Papadopoulos is due to begin serving his term of incarceration on Monday, November 26, 2018.""",
        CitationExtractionResult(
            cases={},
            statutes={
                "28 U.S.C. § 2255": 1,
                "Federal Rule of Civil Procedure 60(b)": 1,
            },
        ),
    ),
    (
        """has failed to demonstrate that the D.C. Circuit is likely to conclude that the appointment of the Special Counsel was unlawful—and, indeed, he has failed even to show that the appeal raises a "close question" that "very well could be decided" against the Special Counsel, id. at 6 (quoting United States v. Quinn, 416 F. Supp. 2d 133, 136 (D.D.C. 2006)). Chief Judge Howell and Judge Friedrich have both issued thorough and carefully reasoned opinions rejecting the arguments that Papadopoulos now champions. See Concord Mgmt. & Consulting LLC, 317 F. Supp. 3d 598 (Friedrich, J.); In re: Grand Jury Investigations, 315 F. Supp. 3d 602 (Howell, C.J.). 

""",
        CitationExtractionResult(
            cases={
                "United States v. Quinn, 416 F. Supp. 2d 133, 136 (D.D.C. 2006)": 1,
                "Concord Mgmt. & Consulting LLC, 317 F. Supp. 3d 598 (Friedrich, J.)": 1,
                "In re: Grand Jury Investigations, 315 F. Supp. 3d 602 (Howell, C.J.)": 1,
            },
            statutes={},
        ),
    ),  # https://casetext.com/case/united-states-v-papadopoulos
]
