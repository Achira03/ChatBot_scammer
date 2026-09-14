// 1) ภาพรวมจำนวน Node ตามชนิด
MATCH (n)
RETURN labels(n)[0] AS type, count(*) AS count
ORDER BY count DESC;

// 2) ภาพรวม ScamType -> วิธี -> สัญญาณ
MATCH p=(s:ScamType)-[:USES_METHOD|HAS_SIGNAL*1..2]->(x)
RETURN p
LIMIT 100;

// 3) วิธีป้องกันของแต่ละ ScamType
MATCH p=(s:ScamType)-[:PREVENTED_BY]->(a:Action)
RETURN p
LIMIT 100;

// 4) Trace กลับไปหลักฐาน
MATCH p=(s:ScamType)-[*1..2]-(e:Evidence)
RETURN p
LIMIT 100;

// 5) ดูกราฟแก๊งคอลเซ็นเตอร์
MATCH p=(s:ScamType)-[*1..2]-(n)
WHERE toLower(s.name) CONTAINS 'คอลเซ็นเตอร์'
RETURN p
LIMIT 100;

// 6) ช่องทางที่ถูกใช้ในการหลอก
MATCH (s)-[:USES_CHANNEL]->(c:Channel)
RETURN labels(s)[0] AS source_type, s.name, c.name, count(*) AS mentions
ORDER BY mentions DESC;

// 7) ข้อมูลที่มิจฉาชีพพยายามขอ
MATCH (s)-[:REQUESTS_INFORMATION]->(i:Information)
RETURN s.name AS source, i.name AS requested_information
LIMIT 100;

// 8) ช่วยตรวจดู orphan semantic nodes
MATCH (n)
WHERE any(x IN labels(n) WHERE x IN ['ScamType','ScamCase','ScamMethod','Signal','Information','Risk','Action'])
  AND NOT (n)--()
RETURN labels(n), n.name
LIMIT 100;
