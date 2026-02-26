import requests
import pandas as pd
import time

URL = "https://www.jamesallen.com/service-api/ja-product-api/diamond/v/2/"

HEADERS = {
    "accept": "application/json, text/plain, */*",
    "content-type": "application/json",
    "origin": "https://www.jamesallen.com",
    "referer": "https://www.jamesallen.com/loose-diamonds/all-diamonds/",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
    "uzlc": "7f9000e09b5389-992e-4b38-a783-872b81058a311-177193217023426825-0028a2c8c50c4a96f141610143963221G9LLJl2174e66d",
    "withcredentials": "true",
    "cookie": "__uzma=e09b5389-992e-4b38-a783-872b81058a31; __uzmb=1771932169; __uzme=7835; __attn_eat_id=77f5e658003f42a2b235cf0d381f39cf; _gcl_au=1.1.81929636.1771932170; __ssds=0; __ssuzjsr0=a9be0cd8e; __uzmaj0=e09b5389-992e-4b38-a783-872b81058a31; __uzmbj0=1771932170; lantern=511a928b-1701-40df-ae1d-d2642a30d4e5; __attentive_id=9cf0cd91e3384793ae933b34730467cd; __attentive_cco=1771932170519; IR_gbd=jamesallen.com; _ga=GA1.1.244000828.1771932171; __attentive_dv=1; QuantumMetricUserID=d4e93042bb6ea2b1b13da3592d1f22b4; gtm_uid=d9fd7645520cf51758c3ed7329627893da43dd63270f552f1f410e3f785c8846; PAPVisitorId=7254sK49QRosmRDvhxWcRCS0Q90vUWBi; OptanonConsent=isGpcEnabled=0&datestamp=Tue+Feb+24+2026+11%3A23%3A16+GMT%2B0000+(Greenwich+Mean+Time)&version=202511.1.0&browserGpcFlag=0&isIABGlobal=false&hosts=&landingPath=NotLandingPage&groups=C0001%3A1%2CC0003%3A1%2CC0002%3A1%2CC0005%3A1%2CC0004%3A1&AwaitingReconsent=false&geolocation=GB%3BENG; forterToken=dce046361d19453a82186509d3e2d51a_1771932197206__UDF43-m4_21ck_m1rKSpZjyyU%3D-1630-v2; _ga_62QHXJXE32=GS2.1.s1771943946$o2$g0$t1771943946$j60$l0$h173944005$dysL8G0JgpE_v6seLC3Xvstde12CiHtW53w; __uzmc=654784951186; __uzmd=1771943962; __uzmf=7f9000e09b5389-992e-4b38-a783-872b81058a311-177193216908811793220-00232e51fab97358c0549; uzmx=7f90008c7bb0b9-995a-4a0f-9423-0a63752a92991-177193216908811793220-d319d89289e535c749"
}

QUERY = "query ($currency: currencies, $isOnSale: Boolean, $sort: sortBy, $lab: [Int] , $price: intRange, $page: pager, $depth: floatRange, $ratio: floatRange, $carat: floatRange, $tableSize: floatRange, $color: intRange, $cut: intRange, $shapeID: [Int], $clarity: intRange, $shippingDays: Int, $isExpressShipping: Boolean, $addBannerPlaceholder: Boolean, $colorIntensityID: [Int]\n    $isFancy: Boolean, $isLabDiamond: Boolean, $polish: [Int], $symmetry: [Int], $flour: [Int], $fancyColorID: Int, $supplierID: Int) {\n    searchByIDs(currency : $currency ,lab: $lab ,isOnSale: $isOnSale, sort: $sort, price: $price, page: $page, depth: $depth, ratio: $ratio, carat: $carat, tableSize: $tableSize, color: $color, cut: $cut, shapeID: $shapeID, clarity: $clarity, shippingDays: $shippingDays, isExpressShipping: $isExpressShipping, addBannerPlaceholder: $addBannerPlaceholder, colorIntensityID: $colorIntensityID, isFancy: $isFancy, isLabDiamond: $isLabDiamond, polish: $polish, symmetry:$symmetry, flour: $flour, fancyColorID: $fancyColorID, supplierID: $supplierID) {\n      hits\n      pageNumber\n      numberOfPages\n      total\n      items {\n        \n    productID\n    sku\n    price\n    stone {\n      isLabDiamond\n      carat\n      depth\n      tableSize\n      shape { id name }\n      color { id name }\n      cut { id name }\n      clarity { id name }\n      lab { id name }\n      flour { id name }\n      symmetry { id name }\n      polish { id name }\n    }\n    \n      }\n    }\n  }\n  "

# Scrape a specific carat range to get coverage across the full price distribution
def build_payload(page_number, is_lab, carat_min, carat_max):
    return {
        "query": QUERY,
        "variables": {
            "price": {"from": 200, "to": 5000000},
            "page": {"count": 4, "size": 50, "number": page_number},
            "depth": {"from": 46, "to": 78},
            "carat": {"from": carat_min, "to": carat_max},
            "tableSize": {"from": 50, "to": 80},
            "color": {"from": 1, "to": 10},
            "cut": {"from": 0, "to": 3},
            "shapeID": [1],
            "clarity": {"from": 1, "to": 9},
            "shippingDays": 999,
            "polish": [4, 3, 2],
            "symmetry": [4, 3, 2],
            "flour": [8, 5, 2, 1],
            "currency": "USD",
            "ratio": {"from": 0.9, "to": 2.75},
            "isLabDiamond": is_lab
        }
    }

def extract_items(raw_items):
    if not raw_items:
        return []
    if isinstance(raw_items[0], list):
        return raw_items[0]
    return raw_items

def scrape_range(is_lab, carat_min, carat_max, max_pages=25):
    all_diamonds = []
    label = "lab" if is_lab else "natural"

    try:
        response = requests.post(URL, headers=HEADERS, json=build_payload(1, is_lab, carat_min, carat_max))
        data = response.json()
        if not data.get("data"):
            return []
        search = data["data"]["searchByIDs"]
        total_pages = min(search["numberOfPages"], max_pages)
        print(f"  {label} {carat_min}-{carat_max}ct: {search['total']} diamonds, scraping {total_pages} pages")
    except Exception as e:
        print(f"  Error on first request: {e}")
        return []

    for page in range(1, total_pages + 1):
        try:
            response = requests.post(URL, headers=HEADERS, json=build_payload(page, is_lab, carat_min, carat_max))
            data = response.json()
            if not data.get("data"):
                break
            raw_items = data["data"]["searchByIDs"]["items"]
            items = extract_items(raw_items)

            for item in items:
                if not isinstance(item, dict):
                    continue
                stone = item.get("stone", {})
                if not stone:
                    continue
                record = {
                    "productID": item.get("productID"),
                    "price_usd": item.get("price"),
                    "is_lab": stone.get("isLabDiamond"),
                    "carat": stone.get("carat"),
                    "depth_pct": stone.get("depth"),
                    "table_pct": stone.get("tableSize"),
                    "color_id": stone.get("color", {}).get("id"),
                    "color_name": stone.get("color", {}).get("name"),
                    "cut_id": stone.get("cut", {}).get("id"),
                    "cut_name": stone.get("cut", {}).get("name"),
                    "clarity_id": stone.get("clarity", {}).get("id"),
                    "clarity_name": stone.get("clarity", {}).get("name"),
                    "lab_cert": stone.get("lab", {}).get("name"),
                    "fluorescence": stone.get("flour", {}).get("name"),
                    "symmetry": stone.get("symmetry", {}).get("name"),
                    "polish": stone.get("polish", {}).get("name"),
                    "shape": stone.get("shape", {}).get("name"),
                }
                all_diamonds.append(record)

            time.sleep(0.3)

        except Exception as e:
            print(f"  Error on page {page}: {e}")
            break

    return all_diamonds

# Scrape across carat ranges to get full price distribution
# Each range gets 25 pages = ~1250 diamonds per range
CARAT_RANGES = [
    (0.3, 0.7),
    (0.7, 1.0),
    (1.0, 1.5),
    (1.5, 2.0),
    (2.0, 3.0),
    (3.0, 5.0),
]

all_diamonds = []

for carat_min, carat_max in CARAT_RANGES:
    print(f"\nScraping {carat_min}-{carat_max} carat range...")
    natural = scrape_range(is_lab=False, carat_min=carat_min, carat_max=carat_max, max_pages=25)
    lab = scrape_range(is_lab=True, carat_min=carat_min, carat_max=carat_max, max_pages=25)
    all_diamonds.extend(natural)
    all_diamonds.extend(lab)
    print(f"  Collected {len(natural)} natural and {len(lab)} lab so far in this range")
    print(f"  Running total: {len(all_diamonds)} diamonds")
    time.sleep(1)

df = pd.DataFrame(all_diamonds)
df = df.drop_duplicates(subset="productID")
df.to_csv("diamonds_raw.csv", index=False)

print(f"\nDone! {len(df)} unique diamonds saved to diamonds_raw.csv")
print(f"Natural: {len(df[df['is_lab']==False])}, Lab: {len(df[df['is_lab']==True])}")
print(f"\nPrice range: ${df['price_usd'].min()} - ${df['price_usd'].max()}")
print(f"Carat range: {df['carat'].min()} - {df['carat'].max()}")
print(df.head())