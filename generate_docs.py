from pathlib import Path
import textwrap
import fitz  # PyMuPDF

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def create_pdf(filename: str, title: str, sections: list[tuple[str, str]]):
    filepath = DATA_DIR / filename
    doc = fitz.open()
    
    width, height = fitz.paper_size("a4")
    margin_x = 54
    margin_top = 54
    margin_bottom = 54
    
    # ── Title Page ──────────────────────────────────────────────────────────
    page = doc.new_page(width=width, height=height)
    page.insert_text((margin_x, 120), title, fontsize=16, fontname="helv", color=(0.1, 0.2, 0.45))
    page.insert_text((margin_x, 145), "FINANCIAL & LEGAL REGULATORY REFERENCE COMPENDIUM", fontsize=9, fontname="helv", color=(0.4, 0.4, 0.4))
    page.insert_text((margin_x, 165), "Official Statutory Text — FinLex AI Document Repository", fontsize=8.5, fontname="helv", color=(0.5, 0.5, 0.5))
    
    y = 210
    page.insert_text((margin_x, y), "Table of Contents & Key Provisions:", fontsize=11, fontname="helv", color=(0.1, 0.1, 0.1))
    y += 22
    for heading, _ in sections:
        page.insert_text((margin_x + 12, y), f"• {heading}", fontsize=9, fontname="helv", color=(0.2, 0.2, 0.2))
        y += 18
        
    # ── Content Pages ───────────────────────────────────────────────────────
    page = doc.new_page(width=width, height=height)
    y = margin_top
    page_num = 2
    
    for heading, content in sections:
        # Check if heading needs a new page
        if y > height - margin_bottom - 60:
            page = doc.new_page(width=width, height=height)
            y = margin_top
            page_num += 1
            
        y += 10
        page.insert_text((margin_x, y), heading, fontsize=11.5, fontname="helv", color=(0.08, 0.22, 0.4))
        y += 18
        
        paragraphs = content.strip().split("\n\n")
        for p in paragraphs:
            lines = textwrap.wrap(p.strip(), width=88)
            for line in lines:
                if y > height - margin_bottom - 20:
                    page = doc.new_page(width=width, height=height)
                    y = margin_top
                    page_num += 1
                    
                page.insert_text((margin_x, y), line, fontsize=9, fontname="helv", color=(0.12, 0.12, 0.12))
                y += 13.5
            y += 8  # paragraph spacing

    doc.save(str(filepath))
    doc.close()
    print(f"Generated {filename}: {page_num} pages at {filepath}")


DPDP_SECTIONS = [
    ("CHAPTER I: PRELIMINARY AND APPLICABILITY", 
     """Section 1: Short title and commencement. This Act may be called the Digital Personal Data Protection Act, 2023 (DPDP Act 2023). It applies to the processing of digital personal data within the territory of India where the personal data is collected in digital form or collected in non-digital form and digitized subsequently. It also applies to processing of digital personal data outside India if such processing is in connection with offering goods or services to Data Principals within India.

Section 2: Definitions. In this Act, unless the context otherwise requires:
(a) 'Data Principal' means the individual to whom the personal data relates and where such individual is a child or a person with disability, includes the parents or lawful guardian.
(b) 'Data Fiduciary' means any person who alone or in conjunction with other persons determines the purpose and means of processing of personal data.
(c) 'Consent Manager' means a person registered with the Board who acts as a single point of contact to enable a Data Principal to give, manage, review and withdraw consent through an accessible, transparent and interoperable platform.
(d) 'Personal Data' means any data about an individual who is identifiable by or in relation to such data.
(e) 'Personal Data Breach' means any unauthorised processing of personal data or accidental disclosure, acquisition, sharing, use, alteration, destruction of or loss of access to personal data, that compromises the confidentiality, integrity or availability of personal data.
(f) 'Significant Data Fiduciary' means any Data Fiduciary or class of Data Fiduciaries as may be notified by the Central Government under Section 10."""),

    ("CHAPTER II: OBLIGATIONS OF DATA FIDUCIARIES",
     """Section 4: Grounds for Processing Personal Data. A person may process personal data of an individual only in accordance with the provisions of this Act and for a lawful purpose for which the Data Principal has given her consent, or for certain legitimate uses as specified in Section 7.

Section 5 & 6: Notice and Consent Requirements. Every request for consent shall be accompanied or preceded by an itemised notice containing clear information on personal data sought, the purpose of processing, the manner in which the Data Principal may exercise her rights of withdrawal and grievance redressal, and how to file a complaint to the Board. Consent must be free, specific, informed, unconditional and unambiguous with a clear affirmative action. The Data Principal shall have the right to withdraw her consent at any time with ease comparable to giving consent.

Section 7: Certain Legitimate Uses. Personal data may be processed without consent for specified purposes including: voluntary submission of data, state benefit disbursement, fulfilling legal obligations under law, compliance with court orders, medical emergency involving threat to life, and employment purposes.

Section 8: General Obligations of Data Fiduciaries. A Data Fiduciary shall implement appropriate technical and organisational measures to ensure compliance with the Act. It shall adopt reasonable security safeguards to prevent personal data breaches. In the event of a personal data breach, the Data Fiduciary shall give the Data Protection Board of India and each affected Data Principal intimation of such breach in such form and manner as may be prescribed. Data must be erased when purpose is served and retention is no longer necessary.

Section 9: Processing Personal Data of Children. A Data Fiduciary shall, before processing any personal data of a child (under 18 years), obtain verifiable consent of the parent or lawful guardian. A Data Fiduciary shall not undertake tracking or behavioral monitoring of children or targeted advertising directed at children.

Section 10: Significant Data Fiduciaries. The Central Government may notify any Data Fiduciary or class of Data Fiduciaries as a Significant Data Fiduciary (SDF) taking into account volume, sensitivity of data, risk to electoral democracy, sovereignty and public order. A Significant Data Fiduciary must appoint an India-based Data Protection Officer (DPO), appoint an independent data auditor to carry out periodic data audits, and undertake Periodic Data Protection Impact Assessments (DPIA)."""),

    ("CHAPTER III: RIGHTS AND DUTIES OF DATA PRINCIPALS",
     """Section 11: Right to Access Information about Personal Data. The Data Principal shall have the right to obtain from the Data Fiduciary a summary of personal data being processed, identities of all other Data Fiduciaries with whom such data has been shared, and any other prescribed information.

Section 12: Right to Correction and Erasure. A Data Principal shall have the right to correction of inaccurate or misleading data, completion of incomplete data, updating of data, and erasure of personal data that is no longer necessary for the purpose of processing, unless retention is required by law.

Section 13: Right of Grievance Redressal. The Data Principal shall have readily available means of grievance redressal provided by the Data Fiduciary or Consent Manager. The Fiduciary must respond within prescribed timelines before the Principal escalates to the Board.

Section 14: Right to Nominate. A Data Principal shall have the right to nominate any other individual to exercise rights on her behalf in event of death or incapacity.

Section 15: Duties of Data Principal. A Data Principal shall not register a false or frivolous grievance or complaint, shall not furnish any false particulars or suppress material information, and shall furnish only verifiable authentic information when exercising rights."""),

    ("CHAPTER V: PENALTIES AND ADJUDICATION",
     """Section 33 & Schedule: Penalties for Non-Compliance. If the Board determines after inquiry that a breach of the provisions has occurred, it may impose financial penalties as specified in the Schedule:
1. Breach in observing reasonable security safeguards to prevent personal data breach under Section 8(5): Financial penalty which may extend up to Two Hundred and Fifty Crore Rupees (₹250,00,00,000 / ₹250 crore).
2. Failure to notify the Board and affected Data Principals of a personal data breach under Section 8(6): Financial penalty which may extend up to Two Hundred Crore Rupees (₹200,00,00,000 / ₹200 crore).
3. Breach of additional obligations in relation to children under Section 9: Penalty up to Two Hundred Crore Rupees (₹200 crore).
4. Breach of obligations of Significant Data Fiduciary under Section 10: Penalty up to One Hundred and Fifty Crore Rupees (₹150 crore).
5. Breach of duties by Data Principal under Section 15: Financial penalty up to Ten Thousand Rupees (₹10,000).""")
]

IBC_SECTIONS = [
    ("CHAPTER I: PRELIMINARY AND SCOPE OF IBC 2016",
     """The Insolvency and Bankruptcy Code, 2016 (IBC 2016) was enacted to consolidate and amend laws relating to reorganization and insolvency resolution of corporate persons, partnership firms and individuals in a time-bound manner for maximization of value of assets, promotion of entrepreneurship, availability of credit, and balancing the interests of all stakeholders.

Key Authorities:
- National Company Law Tribunal (NCLT): Adjudicating Authority for corporate insolvency and liquidation.
- National Company Law Appellate Tribunal (NCLAT): Appellate body over NCLT orders.
- Insolvency and Bankruptcy Board of India (IBBI): The apex regulator overseeing insolvency professionals (IPs), insolvency professional agencies (IPAs), and information utilities (IUs)."""),

    ("CHAPTER II: CORPORATE INSOLVENCY RESOLUTION PROCESS (CIRP)",
     """Section 4 & Threshold: CIRP applies when a corporate debtor commits a default of at least One Crore Rupees (₹1,00,00,000).

Section 7: Initiation by Financial Creditor. A financial creditor (individually or jointly) may file an application for initiating CIRP against a corporate debtor before the NCLT when a default has occurred. The NCLT must ascertain existence of default within 14 days.

Section 8 & 9: Initiation by Operational Creditor. An operational creditor must first deliver a demand notice demanding payment of unpaid operational debt. If the corporate debtor fails to pay or provide notice of existing dispute within 10 days, the operational creditor may file an application under Section 9.

Section 12: Time-Limit for CIRP Completion. CIRP shall be completed within a mandatory period of 180 days from the date of admission. The Committee of Creditors (CoC) may seek a one-time extension of up to 90 days from the NCLT. However, the overall CIRP including any legal proceedings must be completed within 330 days, failing which the corporate debtor proceeds to mandatory liquidation.

Section 14: Moratorium. On the insolvency commencement date, the NCLT declares a moratorium prohibiting:
(a) Institution or continuation of pending suits or proceedings against the corporate debtor.
(b) Transferring, encumbering, alienating or disposing of any assets of the corporate debtor.
(c) Foreclosing, recovering or enforcing any security interest created by the corporate debtor under SARFAESI or otherwise.
(d) Recovery of property by an owner or lessor occupied by the corporate debtor."""),

    ("CHAPTER III: COMMITTEE OF CREDITORS AND RESOLUTION PLAN",
     """Section 21: Committee of Creditors (CoC). The Interim Resolution Professional (IRP) constitutes the CoC consisting solely of financial creditors (excluding related parties). Decisions of the CoC require approval by a voting share of at least 66% of financial creditors for major actions (e.g., appointment of Resolution Professional, extension of time, approval of resolution plan), and 51% for routine administrative matters.

Section 29A: Disqualification of Resolution Applicants. To prevent defaulting promoters from re-acquiring stressed assets at discounted valuations, Section 29A disqualifies persons including:
(a) Undischarged insolvents.
(b) Willful defaulters identified under RBI guidelines.
(c) Promoters or managers of accounts classified as Non-Performing Assets (NPA) for one year or more.
(d) Persons convicted of offences punishable with imprisonment for two years or more.
(e) Disqualified directors under the Companies Act, 2013.

Section 30 & 31: Approval of Resolution Plan. The resolution plan submitted by eligible applicants must provide for payment of insolvency resolution costs in priority, payment to operational creditors not less than liquidation value, and management of affairs of corporate debtor. Once approved by 66% vote of CoC, it is submitted to NCLT for binding approval."""),

    ("CHAPTER IV: LIQUIDATION WATERFALL UNDER SECTION 53",
     """Section 53: Distribution of Assets in Liquidation. In the event of liquidation, the proceeds from the sale of liquidation assets shall be distributed in the following strict order of priority (the 'Waterfall Mechanism'):
1. Priority 1: Insolvency resolution process costs and liquidation costs in full.
2. Priority 2 (Pari Passu): (a) Workmen's dues for the period of 24 months preceding the liquidation commencement date; and (b) Debts owed to a secured creditor who has relinquished security interest to the liquidation estate.
3. Priority 3: Wages and any unpaid dues owed to employees other than workmen for the period of 12 months preceding the liquidation commencement date.
4. Priority 4: Financial debts owed to unsecured creditors.
5. Priority 5 (Pari Passu): (a) Dues to the Central Government and State Government (taxes) for the period of 2 years; and (b) Balance debt owed to secured creditors who enforced security outside liquidation.
6. Priority 6: Any remaining debts and dues (including trade operational debts).
7. Priority 7: Preference shareholders, if any.
8. Priority 8: Equity shareholders or partners.""")
]

FEMA_SECTIONS = [
    ("CHAPTER I: OBJECTIVES AND SCOPE OF FEMA 1999",
     """The Foreign Exchange Management Act, 1999 (FEMA 1999) replaced the earlier Foreign Exchange Regulation Act (FERA) to facilitate external trade and payments and promote the orderly development and maintenance of the foreign exchange market in India. Violations under FEMA are civil contraventions rather than criminal offenses.

Section 2: Key Concepts:
(a) 'Capital Account Transaction': A transaction that alters the assets or liabilities, including contingent liabilities, outside India of persons resident in India or assets/liabilities in India of persons resident outside India.
(b) 'Current Account Transaction': A transaction other than a capital account transaction, including payments due in connection with foreign trade, current business, services, normal short-term banking and credit facilities, interest on loans, and net income from investments.
(c) 'Person Resident in India (PRI)': An individual residing in India for more than 182 days during the course of the preceding financial year, subject to conditions regarding employment, business, or intention to stay for an uncertain period."""),

    ("CHAPTER II: REGULATION AND MANAGEMENT OF FOREIGN EXCHANGE",
     """Section 3: Dealings in Foreign Exchange. No person shall deal in or transfer foreign exchange or foreign security to any person other than an authorised person (AP), make payment to or for the credit of any person resident outside India, or receive payment without general or special permission of the Reserve Bank of India (RBI).

Section 5: Current Account Transactions & LRS. Any person may sell or draw foreign exchange to or from an authorised person if such transaction is a current account transaction. The Central Government, in consultation with RBI, imposes reasonable restrictions under the Foreign Exchange Management (Current Account Transactions) Rules, 2000.
Liberalised Remittance Scheme (LRS): Resident individuals are permitted to freely remit up to USD 250,000 (United States Dollars Two Hundred and Fifty Thousand) per financial year for permissible current or capital account transactions, including overseas education, travel, medical treatment, gift/donation, and investment in shares/property overseas.

Section 6: Capital Account Transactions & FDI/ECB. Capital account transactions are prohibited unless specifically permitted by RBI regulations. Major routes include:
- Foreign Direct Investment (FDI): Permitted under Automatic Route (no prior government or RBI approval required) and Government Route (approval required from relevant Ministry).
- External Commercial Borrowings (ECB): Commercial loans raised by eligible resident entities from recognized non-resident entities, subject to parameters such as minimum average maturity period (MAMP) of 3 to 5 years, all-in-cost ceiling (SOFR benchmark + margin), and negative list end-uses (real estate, equity investment)."""),

    ("CHAPTER IV: CONTRAVENTIONS, PENALTIES AND COMPOUNDING",
     """Section 13: Penalties for Contraventions. If any person contravenes any provision of FEMA, or any rule, regulation, notification or order, he shall be liable to a penalty up to:
(a) Three times the sum involved in such contravention where the amount is quantifiable; or
(b) Up to Two Lakh Rupees (₹2,00,00), where the amount is not quantifiable;
(c) In case of continuing contravention, an additional penalty which may extend to Five Thousand Rupees for every day after the first day during which the contravention continues.

Section 15: Compounding of Contraventions. Contraventions under Section 13 may be compounded by designated officers of the Reserve Bank of India (or Directorate of Enforcement for certain cases) within 180 days from receipt of application. Once compounded, no further proceedings or penalty can be initiated for that contravention.

Section 37: Directorate of Enforcement (ED). The officers of Enforcement Directorate are empowered to investigate contraventions, inspect accounts, summon witnesses, search and seize assets under Section 37.""")
]

def main():
    print("Generating statutory and regulatory PDFs...")
    create_pdf("dpdp2023.pdf", "DIGITAL PERSONAL DATA PROTECTION ACT, 2023", DPDP_SECTIONS)
    create_pdf("ibc2016.pdf", "INSOLVENCY AND BANKRUPTCY CODE, 2016", IBC_SECTIONS)
    create_pdf("fema1999.pdf", "FOREIGN EXCHANGE MANAGEMENT ACT, 1999", FEMA_SECTIONS)
    print("All documents generated successfully in data/.")

if __name__ == "__main__":
    main()
