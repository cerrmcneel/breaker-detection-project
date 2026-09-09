import hashlib
import os
import time
import urllib.error
import urllib.request

# Directory to save raw images for user review
save_dir = r"C:\ironhack\labs\breaker-detection-project\data\scraped_raw"
os.makedirs(save_dir, exist_ok=True)

# List of unique image URLs extracted from visual search results
urls = [
    "https://i.pinimg.com/736x/51/c3/21/51c3215ca23a78ac1b7f7b1db306dbb5.jpg",
    "https://i.pinimg.com/736x/24/fd/49/24fd4929d53f9f06b7cefbac757e2fc6.jpg",
    "https://5.imimg.com/data5/SELLER/Default/2025/2/487332934/KR/AJ/DQ/9431399/industrial-cabling-wiring-services-500x500.jpeg",
    "https://www.hamuch.com/trade/server/php/files/IMAG0094.jpg",
    "https://lookaside.fbsbx.com/lookaside/crawler/media/?media_id=100067583956393",
    "https://img3.exportersindia.com/product_images/bc-full/dir_125/3720269/molded-case-circuit-breaker-2031764.jpg",
    "https://i.pinimg.com/736x/8b/e1/b3/8be1b35c40f8a2468c4b8fa308868559.jpg",
    "https://stewart-switch.com/pictures/jNkfU1CjMPCxFrnwbkUPXDylhY1GiDOUAoDCungs002lwPH106_JcoHhoM8Tq4NFTGnRFR4jkhgK15nSX94lEMyYWdPqH700FLFc2TowLsYgu88VHAzk6hqw8m_5pyuPw1200-h630-p-k-no-nu.jpg",
    "https://5.imimg.com/data5/SELLER/Default/2024/2/390315513/GV/KR/KE/180786463/electric-control-panel-1000x1000.jpg",
    "https://5.imimg.com/data5/SELLER/Default/2021/3/SS/GH/ER/38171117/solar-acdb-500x500.jpg",
    "https://www.diynot.com/diy/media/untitled.42466/full",
    "https://affinityelectrical.co.uk/wp-content/uploads/opt_D3.jpg",
    "https://i.pinimg.com/736x/4c/b7/b1/4cb7b1a64663d8a5a913d0df04a9b849.jpg",
    "https://www.bradleystokeelectrical.co.uk/wp-content/uploads/2017/11/Commercial-Link-6-Distribution-board-change-510x382.jpg",
    "https://i.pinimg.com/736x/60/c8/70/60c870a36ee350f09d24a524b95e06c2.jpg",
    "https://i.pinimg.com/736x/5d/da/0a/5dda0acbef469ba470e90c4b8510d592.jpg",
    "https://www.mrpowerguy.com/assets/images/upimages/s-07-20220111052246.jpg",
    "https://www.fuseboxdiagram.net/wp-content/uploads/2023/03/house-fuse-box-wiring-diagram-1024x768.jpg",
    "https://i.pinimg.com/236x/94/98/74/949874234b79ad109d348cf83373a4a7.jpg",
    "https://i.pinimg.com/736x/37/5b/db/375bdb0d240546935effcf78f7132e79.jpg",
    "https://5.imimg.com/data5/ANDROID/Default/2022/12/IS/WB/NR/10861879/product-jpeg-500x500.jpg",
    "https://media.karousell.com/media/photos/products/2020/7/26/electrical_services_1595762551_0b2399cf.jpg",
    "https://community.iotawatt.com/uploads/db6340/optimized/2X/3/321fe576ca6017d87b6b979edccd57ae4cfb9483_2_690x494.jpeg",
    "https://img2.exportersindia.com/product_images/bc-full/2022/6/9691204/mcb-distribution-boards-1655707757-6403250.jpeg",
    "https://media.karousell.com/media/photos/products/2022/11/15/electrical_wiring_socket_1668527499_879cfa5d.jpg",
    "https://thumbs.dreamstime.com/b/automatic-switches-wires-electrical-shield-close-up-electrical-shield-automatic-switches-electricity-house-172664325.jpg",
    "https://www.helsdownelectrical.co.uk/wp-content/uploads/2019/04/20190326_162159.jpg",
    "https://5.imimg.com/data5/SELLER/Default/2023/5/305028744/EN/QU/RA/3963877/panel-500x500.JPG",
    "https://alfazalindustry.com/wp-content/uploads/2022/05/WhatsApp-Image-2022-05-14-at-11.01.25-AM-2.jpeg",
    "https://opik.fyysika.ee/media/img/exp/id-234842.v-0.w-1400.jpg",
    "https://i.pinimg.com/originals/52/74/c6/5274c63284f7b669c83a470325e12996.jpg",
    "https://i.redd.it/n5ojpzs6l2601.jpg",
    "https://i.redd.it/fmhfd5rfci0c1.jpg",
    "https://img3.exportersindia.com/product_images/bc-full/2022/6/9691204/mcb-distribution-boards-1655707757-6403250.jpeg", # Duplicate in raw list, we'll deduplicate
    "https://img3.exportersindia.com/product_images/bc-full/dir_157/4693929/light-distribution-board-1528111336-3941882.png",
    "https://5.imimg.com/data5/GLADMIN/Default/2022/11/JC/HW/PF/96537/youtube-1000x1000.jpg",
    "https://i.pinimg.com/736x/0c/92/d2/0c92d27a03e72961f8f6b7233a65e098.jpg",
    "https://i.originals/dc/72/22/dc7222668afd4d096152c67402d09513.jpg",
    "https://www.fuseboxdiagram.net/wp-content/uploads/2023/03/house-fuse-box-wiring-diagram-768x576.jpg",
    "https://r2.community.samsung.com/t5/image/serverpage/image-id/3683146i6CF7EC5497951433?v=v2",
    "https://5.imimg.com/data5/SELLER/Default/2025/1/478082282/YS/HB/NX/127907580/electrical-switchgear-control-panel-1-500x500.jpg",
    "https://5.imimg.com/data5/SELLER/Default/2021/3/SS/GH/ER/38171117/solar-acdb-1000x1000.jpg",
    "https://copperroadelectrical.com.au/wp-content/uploads/2021/09/cre-gallery-11.jpg",
    "https://i.pinimg.com/736x/01/f7/1e/01f71e066d20583718319a37d3fed9ab.jpg",
    "https://sc04.alicdn.com/kf/U38f0ccdaf5cf48ca9fb6b9acdbe0a1ccj/951539058/U38f0ccdaf5cf48ca9fb6b9acdbe0a1ccj.jpg",
    "https://3.imimg.com/data3/LF/FR/MY-11536672/distribution-board.jpg",
    "https://5.imimg.com/data5/SELLER/Default/2021/11/BT/HP/RN/6821487/electrical-distribution-box-1000x1000.jpg",
    "https://i.pinimg.com/originals/a9/8f/95/a98f956a0fec4fc92b3ebf0687ab4a08.jpg",
    "https://5.imimg.com/data5/SELLER/Default/2022/8/WK/GY/KF/97395576/solar-dc-distribution-box-500x500.jpg",
    "https://5.imimg.com/data5/SELLER/Default/2024/4/411424272/UL/XS/LA/95352013/electrical-wiring-services-500x500.png",
    "https://5.imimg.com/data5/SELLER/Default/2022/1/PY/HR/BU/25794863/solar-dcdb-30-kw-6-in-6-out-500x500.jpg",
    "https://i2.wp.com/content3.jdmagicbox.com/comp/gurgaon/p1/011pxx11.xx11.160706132356.k8p1/catalogue/ajay-electrical-works-dlf-city-phase-5-gurgaon-electricians-tm6uk0d51c.jpg?strip=all",
    "https://i.pinimg.com/originals/6c/0f/68/6c0f688058a22951e8c4895ca6d63506.jpg",
    "https://www.wirral-electrician.co.uk/images/whychosegallery/jj.jpg",
    "https://i.pinimg.com/originals/cf/ff/22/cfff2215507df2b95d87c062d2004f6b.jpg",
    "https://naatechnical.com/wp-content/uploads/2024/05/WhatsApp-Image-2024-05-04-at-21.11.07_c7f3527c.jpg",
    "https://preview.redd.it/my-1st-electrical-panel-v0-e8pworlubj5b1.jpg?width=640&crop=smart&auto=webp&s=3fc1f0a1a4859b23468753d13ff92a8c47a04664",
    "https://i.pinimg.com/236x/f9/ff/f4/f9fff4b19f78a612e3ed733acd54dd86--robotics-circuit.jpg",
    "https://lookaside.fbsbx.com/lookaside/crawler/media/?media_id=3964750646880010",
    "https://tiimg.tistatic.com/fp/1/007/908/electric-switch-board-109.jpg",
    "https://lookaside.fbsbx.com/lookaside/crawler/media/?media_id=100063989236740",
    "https://i.pinimg.com/736x/ab/a5/42/aba542713cb482590f4b21ff843466b6.jpg",
    "https://myconstructor.co.uk/assets/images/professionals/10491/profile1678322679.png",
    "https://media.karousell.com/media/photos/products/2022/7/12/electrical_1657609849_a365372a_progressive.jpg",
    "https://5.imimg.com/data5/SELLER/Default/2021/1/DV/ME/SO/101606276/distribution-panels-500x500.jpg",
    "https://5.imimg.com/data5/SELLER/Default/2023/11/358210616/ZQ/ER/SG/11087882/electrical-industrial-wiring-500x500.jpg",
    "https://i.pinimg.com/736x/25/68/ab/2568ab79e70ba6f7dae60297d9d9887f.jpg",
    "https://acen-solar.co.uk/wp-content/uploads/2023/03/Domestic-14-Way-Upgrade-London.jpg",
    "https://5.imimg.com/data5/SELLER/Default/2022/1/PY/HR/BU/25794863/solar-dcdb-30-kw-6-in-6-out-250x250.jpg",
    "https://lookaside.fbsbx.com/lookaside/crawler/media/?media_id=100064759024288",
    "https://i.pinimg.com/originals/f3/33/05/f33305bdf2168abda66543ea1545fd67.jpg",
    "https://i.pinimg.com/236x/e9/48/11/e94811dcffb15a7b80b12b3ac910fffd--rewiring-a-house-refrigerator.jpg",
    "https://i.pinimg.com/736x/89/27/8c/89278cb708b4e613cca6a9c10bfb2624.jpg",
    "https://i.sstatic.net/TlbnI.jpg",
    "https://forums.autodesk.com/t5/image/serverpage/image-id/387019iC5A300A22B61B7BF?v=v2",
    "https://i.pinimg.com/736x/04/0a/d7/040ad7dc60c4f7ead618a843dfc055d7.jpg",
    "https://www.prolectrical.co.uk/wp-content/uploads/2018/07/IMG-20180109-WA0001-1-800x600.jpg",
    "https://i.pinimg.com/736x/7c/9a/f2/7c9af2b25d8a13d611874d3f398c6ef4.jpg",
    "https://i.pinimg.com/200x150/51/c3/21/51c3215ca23a78ac1b7f7b1db306dbb5.jpg",
    "https://i.pinimg.com/originals/5f/20/3e/5f203ebe16aefacff19e5073242639cc.jpg",
    "https://i.pinimg.com/originals/af/93/e6/af93e665ea3a90c75b2a0875c03917da.jpg",
    "https://www.diynot.com/diy/media/untitled.69500/full?d=1385220241",
    "https://5.imimg.com/data5/ANDROID/Default/2022/9/ZZ/EE/FB/2531278/product-jpeg-500x500.jpg",
    "https://i.pinimg.com/736x/1d/b7/dc/1db7dc7afcb015308cf7fb60d0059973.jpg",
    "https://i.pinimg.com/736x/32/2f/69/322f69e45164f8bc089463cc7a861a6f.jpg",
    "https://i.pinimg.com/originals/60/f1/72/60f1728b3e8a104217bf60488fe90a7c.jpg",
    "https://i.pinimg.com/236x/5f/98/e0/5f98e01a84d42e8f8016f8d20032f260.jpg?nii=t",
    "https://i.pinimg.com/236x/3f/1e/92/3f1e92e843202b179870240923ec39f9.jpg",
    "https://static.professional-electrician.com/professional-electrician/uploads/2016/01/20151014_123113-1024x576.jpg",
    "https://i.pinimg.com/originals/d6/8d/71/d68d719258c78b80f80077c939164ab9.jpg",
    "https://i.pinimg.com/736x/73/f9/47/73f947a75c7d7f93031e727f0c9fad9e--electrical-projects-electrical-wiring.jpg",
    "https://i.pinimg.com/200x150/96/f5/62/96f562c0e5e1d5d1af074b8b82debe32.jpg",
    "https://i.pinimg.com/736x/d3/8e/98/d38e989e28c91a147ea87033732df4a0.jpg",
    "https://media.licdn.com/dms/image/v2/C4E22AQEVfZZ_dFzG4A/feedshare-shrink_2048_1536/feedshare-shrink_2048_1536/0/1664552060582?e=2147483647&v=beta&t=5ZZhsr7bk5VphmGGoQQ3AWR0_vuU5curCMVWfyt9GWc",
    "https://i.pinimg.com/originals/e6/c3/ec/e6c3ec1722bd11c08e11404cb08b774a.jpg",
    "https://i.ytimg.com/vi/RJy9rVK1hug/maxresdefault.jpg",
    "https://5.imimg.com/data5/SELLER/Default/2021/11/GJ/BO/IO/88030706/solar-dc-distribution-box-4-in-4-out-with-dc-spd-dc-mcb-500x500.jpg",
    "https://assets.aseannow.com/forum/uploads/monthly_2019_09/IMG_8107.JPG.335c38e4da76d8c61b14f04e83b2cfea.JPG",
    "https://media.licdn.com/dms/image/C4E22AQEaXvzbsEMFvg/feedshare-shrink_800/0/1565298658746?e=2147483647&v=beta&t=I1YbRBNjyLyGV-YX9rygpsGTiA0dI98wlsf6_VQF-M0",
    "https://i.pinimg.com/736x/22/c7/b7/22c7b7c7dbd69350d200eca356f8886f.jpg",
    "https://i.pinimg.com/736x/ea/2c/45/ea2c45f454b89877d331950754896144.jpg",
    "https://i.pinimg.com/originals/90/66/89/9066897c951b4d64ca304ecb36959b70.jpg",
    "https://www.daylightelectrician.com/wp-content/uploads/2017/09/db-box-distribution-board-Change-mcb-63A-main-circuit-breaker-electrician-singapore-landed-joo-chiat-1_wm-800x500.jpeg",
    "https://i1.wp.com/5.imimg.com/data5/LG/UX/OE/SELLER-55492890/mcb-mccb-connection-service-500x500.png?strip=all",
    "https://5.imimg.com/data5/SELLER/Default/2022/11/MJ/JP/GL/956860/wire-rope-hoist-500x500.jpeg",
    "https://i.pinimg.com/736x/41/82/56/41825670c7a17e8cfd5c4e5cde7b748f.jpg",
    "https://5.imimg.com/data5/SELLER/Default/2024/10/460390084/TB/IM/GJ/208855497/ht-and-lt-electrical-works-500x500.jpeg",
    "https://i.pinimg.com/736x/fa/4b/9d/fa4b9d35bd9db68fcb1651085382ced6.jpg",
    "https://i.pinimg.com/originals/a8/86/fb/a886fbeb1cee2529d9a8802e0ca806e7.jpg",
    "https://i.ytimg.com/vi/UswbkQILkU0/hqdefault.jpg",
    "https://i.pinimg.com/originals/45/ce/01/45ce01e7acdc465892c8e2fe75a1f19c.jpg",
    "https://5.imimg.com/data5/NSDMERP/Default/2024/9/452159295/MV/ZC/PC/30174700/30174700-product-1726748948908-250x250.jpg",
    "https://i.pinimg.com/originals/d3/52/20/d3522074da53195596e6a91c07c26a05.jpg",
    "https://allwiringsketch.com/wp-content/images/Electrical-Fuse-Box-Wiring-1823.jpg",
    "https://i.pinimg.com/236x/36/31/42/3631427450edf8b5ae32e2f4d7956b67.jpg",
    "https://i.pinimg.com/236x/c9/ad/24/c9ad24e7a4d9207478e8fbf8dea00baa.jpg",
    "https://thumbs.dreamstime.com/b/electrical-engineer-work-called-db-dressing-352867406.jpg",
    "https://i.pinimg.com/originals/e9/f9/7e/e9f97e43b30410f0764b989630142c06.jpg",
    "https://i.pinimg.com/originals/c1/f9/4b/c1f94b9fbd01e3e0080021a325d63ddb.jpg",
    "https://4.imimg.com/data4/YR/BB/MY-23042982/transformer-oil-filter-machine-servicing-1000x1000.jpg",
    "https://i.pinimg.com/736x/00/23/09/002309e917e4a93ce78991e423322e70.jpg",
    "https://wireblueprint.com/wp-content/pic/kmi_Electrical_Wiring_Practice_Board.jpg",
    "https://i.pinimg.com/originals/fb/0c/cb/fb0ccbfff000eb4c43f5f55b1c14ae16.jpg",
    "https://media.karousell.com/media/photos/products/2022/9/4/electrical_wiring_1662273297_17f1265c_progressive.jpg",
    "https://i.pinimg.com/736x/90/b3/5e/90b35e26186a207946c3801ff02d36fe.jpg",
    "https://5.imimg.com/data5/SELLER/Default/2025/1/477652358/QK/IC/NE/155776237/three-phase-acdb-32a-with-ryb-sibass-1000x1000.png",
    "https://i.pinimg.com/474x/41/bb/94/41bb9455676e6c09a6abc9fe7cfcea91.jpg",
    "https://i.pinimg.com/originals/64/45/3f/64453f2340a4da6b7a2523da8a411d3e.jpg",
    "https://i.pinimg.com/736x/ed/bd/2b/edbd2b6c2e7f2d6d546a4d962a11045b.jpg",
    "https://5.imimg.com/data5/SELLER/Default/2021/12/GZ/WZ/AO/77711013/lt-mcb-box-500x500.jpg",
    "https://media.karousell.com/media/photos/products/2019/04/06/perkhidmatan_wiring_dan_aircon_1554557403_4d150f05.jpg",
    "https://i.pinimg.com/170x/b1/f8/a1/b1f8a1246710216549c7d916b341c9b2.jpg",
    "https://huntelectricalltd.com/wp-content/uploads/2023/03/Electricians-in-Suffolk-800x1200-1.jpg",
    "https://i.pinimg.com/736x/b3/85/67/b38567f2a932bd5bdfe61166b8de8056.jpg",
    "https://srelectrical.co.za/wp-content/uploads/2023/06/22-1.jpg",
    "https://images.squarespace-cdn.com/content/v1/63b432f97c892c4328f77e14/f7a2b604-0221-4778-99d7-31ff45d39cb2/Screenshot+2023-01-23+100517.jpg",
    "https://3.imimg.com/data3/SK/YY/MY-3782815/lighting-distribution-panel-1000x1000.jpeg",
    "https://proteusswitchgear.com/wp-content/uploads/2021/04/3-phase-distribution-boards-640x857.jpg",
    "https://i.ytimg.com/vi/T4b4qSwHZjk/oar2.jpg?sqp=-oaymwEkCJUDENAFSFqQAgHyq4qpAxMIARUAAAAAJQAAyEI9AICiQ3gB&rs=AOn4CLCfcv5lLmVR4HoRP8n0MZN_wjn4Cw",
    "https://5.imimg.com/data5/SELLER/Default/2022/5/GX/YG/BH/142810558/11-500x500.png",
    "https://5.imimg.com/data5/SELLER/Default/2023/12/369134823/BP/IS/BG/4454953/final-distribution-board-500x500.jpg",
    "https://www.truck1.eu/img/Construction_machinery_Construction_equipment_HAMM_REFULER_poglebiarka_ssaca_GULLIES_suction_dredge_OVRAGI_vsas-ful-9027/9027_7860418840582.jpg",
    "https://i.pinimg.com/736x/d4/86/61/d48661db811c0f66c3868e1a1e49f3b3.jpg",
    "https://assets.sc.hager.com/uk/-/media/project/hagerdeep/united-kingdom/hager/b2b/get-involved/tool-up/invicta-install-4.jpg?w=440&hash=40C22020943C4BBAE738685CF041688D",
    "https://media.karousell.com/media/photos/products/2023/7/18/db_metal_clad_3_fasa_1689689040_12c90cab_progressive.jpg",
    "https://5.imimg.com/data5/SELLER/Default/2023/11/362831049/JQ/NA/PX/130191585/telephone-wiring-service-500x500.jpeg",
    "https://i.pinimg.com/originals/8a/b0/f2/8ab0f280de05e1bc0c9aadeb3dc734f1.jpg",
    "https://i.pinimg.com/originals/60/9e/22/609e22bfd817194bae79fd08a67ea02a.jpg?nii=t",
    "https://i.pinimg.com/originals/ce/2f/6c/ce2f6c194e090625b1c739d1eaa3416f.jpg",
    "https://5.imimg.com/data5/SELLER/Default/2023/8/334501592/XQ/GC/QQ/31248557/internal-electrification-works-500x500.jpeg",
    "https://www.homehow.co.uk/images/fuseboard2.jpg",
    "https://i.pinimg.com/736x/b0/b5/da/b0b5da3eaecd4e82086f91e3e6e47880.jpg",
    "https://5.imimg.com/data5/SELLER/Default/2024/10/460390084/TB/IM/GJ/208855497/ht-and-lt-electrical-works-1000x1000.jpeg",
    "https://media.karousell.com/media/photos/products/2023/11/21/electrician__electricial_servi_1700540998_fe94dc33_progressive.jpg",
    "https://5.imimg.com/data5/UF/MU/GLADMIN-66876489/power-distribution-panels-500x500.jpg",
    "https://5.imimg.com/data5/IOS/Default/2024/10/460530554/MS/ZQ/DQ/2603838/product-jpeg-1000x1000.png",
    "https://i.pinimg.com/736x/3f/5f/5b/3f5f5be7ee82a280853e13833e48b844.jpg",
    "https://image.made-in-china.com/2f0j00nQybNhtKLzgA/Consumer-Control-Unit-Complete-with-Accessories-Distribution-Board-.jpg",
    "https://5.imimg.com/data5/SELLER/Default/2021/12/LX/AQ/YK/18828709/electrical-contractor-services-1000x1000.jpeg",
    "https://5.imimg.com/data5/SELLER/Default/2023/9/345898921/MD/JP/CJ/14440844/distribution-energy-audit-services-500x500.jpeg",
    "https://5.imimg.com/data5/SELLER/Default/2023/11/362830925/SW/BP/WY/130191585/telephone-wiring-service-500x500.jpeg",
    "https://i.pinimg.com/736x/fd/01/9e/fd019e9017d1184ab4f52eace082b864.jpg",
    "https://lookaside.fbsbx.com/lookaside/crawler/media/?media_id=1254150896731156",
    "https://5.imimg.com/data5/ANDROID/Default/2022/9/EE/KW/EK/35988005/product-jpeg-500x500.jpg",
    "https://i.pinimg.com/736x/25/d3/50/25d35042b6e79aad0b0d94e217707c83.jpg",
    "https://5.imimg.com/data5/SELLER/Default/2020/11/QX/OR/SS/14015443/20150601-202115-1000x1000.jpg",
    "https://i.pinimg.com/736x/34/9e/4f/349e4fac5e45c7b0795e99adfdb186fc.jpg",
    "https://www.duijndam-machines.com/images/machines/drilling-station-for-drilling-holes-into-filled-trays-105124.jpg?quality=85&type=webp",
    "https://i.redd.it/2perolmdlx351.jpg",
    "https://tiimg.tistatic.com/fp/1/003/501/plc-service-063.jpg",
    "https://5.imimg.com/data5/SELLER/Default/2023/6/318813175/QL/AS/EH/11738036/lt-distribution-box-500x500.jpg",
    "https://threebestrated.in/images/MANZOORElectrician-Moradabad-UP.jpeg",
    "https://5.imimg.com/data5/SELLER/Default/2023/8/331642990/HP/GV/AF/1200987/control-panel-board-500x500.jpeg",
    "https://img1.exportersindia.com/product_images/bc-full/2020/5/4630261/industrial-electrical-panel-1588849242-5416117.jpeg",
    "https://es.habcdn.com/photos/business/medium/cuadro-electrico-para-vivienda-1452653.jpg",
    "https://es.habcdn.com/photos/business/medium/cuadro-general-de-mando_260361.jpg",
    "https://electricidadiem.es/wp-content/uploads/2013/04/cuadro-01.jpg",
    "https://forumdacasa.com/extensions/InlineImages/image.php?AttachmentID=144657",
    "https://thf.bing.com/th/id/R.dc81ead1882f98653cd8ea2ca3d770af?rik=gA9AUekh4j2cRQ&riu=http%3a%2f%2fbouw.claesnet.eu%2fwp-content%2fuploads%2f2011%2f11%2fIMAG0639.jpg&ehk=qY%2f7asL742fn%2fFvLCCaPDOfRk2ZAYTJK3vRlZAZYM%2bQ%3d&risl=&pid=ImgRaw&r=0",
    "https://www.elektro-zavodnik.si/assets/content_album_pic/0263177001573822911.jpg",
    "https://cs8.pikabu.ru/post_img/big/2017/01/01/10/14832926861535585.jpg"
]

# Deduplicate URL list first
unique_urls = list(dict.fromkeys(urls))

timestamp = int(time.time())
downloaded_count = 0
errors = []

print(f"Starting download of {len(unique_urls)} unique Bing Visual Search images...")

for index, url in enumerate(unique_urls):
    # Extract extension or fallback to jpg
    ext = ".jpg"
    if ".png" in url.lower():
        ext = ".png"
    elif ".webp" in url.lower():
        ext = ".webp"
        
    # Stable name with URL hash to prevent duplicates and enable resume
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()[:12]
    filename = f"bing_visual_{url_hash}{ext}"
    filepath = os.path.join(save_dir, filename)
    
    if os.path.exists(filepath):
        print(f"  [{index+1}/{len(unique_urls)}] Already exists: {filename}")
        downloaded_count += 1
        continue
        
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=8) as response:
            # Check content type
            info = response.info()
            content_type = info.get_content_type()
            if "image" not in content_type:
                continue
                
            with open(filepath, 'wb') as out_file:
                out_file.write(response.read())
        downloaded_count += 1
        print(f"  [{index+1}/{len(unique_urls)}] Downloaded: {filename}")
        time.sleep(0.1) # Be gentle
    except Exception:
        # Silent pass or debug logs
        pass

print("\nDownload completed.")
print(f"Total processed/downloaded images: {downloaded_count}/{len(unique_urls)}")
