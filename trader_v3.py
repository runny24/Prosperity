# ─────────────────────────────────────────────────────────────────────────────
# trader.py  —  Prosperity 4 Round 5  |  Neural Market-Making Bot  v3
#
# Model      : Distilled tiny MLP (teacher IC=0.34, 175k-row MemorizingMixtureAlphaNet)
# Parameters : 4,409  (float16, hex-encoded — only numpy + json imported)
# Strategy   : maker-only skewed quotes + active position flattening
#
# v3 vs v2 fixes:
#   - gzip / base64 / struct removed (not in Prosperity sandbox allowed list)
#   - Weights encoded as plain hex strings, decoded with bytes.fromhex() (builtin)
#   - Distilled from 175k-row full model (teacher IC=0.34 vs actual returns)
# ─────────────────────────────────────────────────────────────────────────────
from datamodel import OrderDepth, TradingState, Order
import numpy as np
import json

# ── Weight blob (float16, hex-encoded) ───────────────────────────────────────
_W_HEX = {
    "sc_center": "80490000068000000000000000000000000000001194c1c10000000000000000003a00000080f2b4",
    "sc_scale": "0045003c0600003c813f3640f8541059003c003c9159ec5b003c003c9c59003c00340040b43c793e",
    "emb": "c823a4aace28082a3cae2caca61e0fa4ef9469a9a129ff282a1b352b0d2682a9b0241caa9d29b698afa057a8d82bf39f7ba585a4cb26bc244f259425db2e33a58fa632a9dc27402bce9d5aa943a92a28c128f82643a8fa21081e9a219729b8215c28c21a51231028aaac0ea405a616a80ba3c82c90a9362a63ad3aa4322084278f280a2984a4dca57f29b3a016ac41abdfa94e26061f722882a53f20fd27e4a3df253ca8a9a4d1a4a5ab402605a635113626b5253f28b39c71a5fe9b66ada2ac39ac60aa0725c5a9e3a42b2c95a4bd28f6a956a11daa37a45b297ead8e25222292a7a7ac951ab1a6d72136171f2d20a28aaa9f208f215b2c54299124b02c9dab16a8eca907a565ac1028a2a1b4ac80aa05246029622eab30512550aad1aed62232a94ea5e2a7f19ed22a2e20fa9fa2aa5ead2ca6f699ac2d8d1e562a5aa691ab27aaa8a8cd2e3aa95b29c7a9f729e724df9f779837a8bb25d9a5d5ab18ac6e200d2df31f4d9e6e232c2cd9214da5192b7825e52c9420cfab63ac502d812d87a0dea4aa2c1eabada91d28f4a4ff8a3fa04724882ca31c23288cae82995ea9d0ab9ea94ca4da293dac87162aa631a4692ae126472cef9f6fa8a7a149228920f0aa961cde210730042e75a949a3adac90a70024bb2bafa827a8deaef79fc8aac4a699a9df9ca1adadacd12e1cac04abe02da63082abbea46924872d49aa48a2fb289925dda4079f61ac17a2a9a2ee211ba70c26732ab21f902452198d0f9ea80c24e59cac2e042dff24e6a80fab9b2bc522282695adef29dc2513ab09a6c82213196ea3749d82a20920b62c1ab013a28020a523c4a4a2ad4824872de4ac8a2938a4e699c3aa51a42b2c71278e2d872ba025952d8525442976a3b21a64a8532a8aab0029c8a07f2bcf28d52da72806ad039b7faba6a6d62c1e2871a878aaca16cbab49a8ff23632bd62754afd3237c1b4b252d2d9aae93a79c270dadba2cba2469295da73d29d42a1b220c220d2381a9829fe39ff524692ac6aa3c9ebb2cbfa846a91e216c295c11422c822961291d27722f40a16aae242dd6a5b21d312c21ad02ac25ad0ca9a225c9262a2383ac62af96190da6da257d28501fe228fa21c62681a8",
    "W1": "14aabc2a0733b2ae71308caca9a7123344b88ab553b6fdb8d1b17632d9ac862aa534aa35aea4c335572623b313b147b696b321aed5325f37ecb5e52e72b412307ba8d2b7ef3332b197b08fb492b13b2fc8aabe2d322e9f35d3ad48b68934abadc434962e11a8f12d28387038e624aa299b3447b36e2db79c682d1fb3c8b66fb8c7b4d8af04ad2438bb341db4f32df6aeddac572a4c31cead8d34e2b4f6168e39a22b21364326f830c4a3d4a9abb25c2f48b3b131a434452ee5acc42f48ad0034d5b7bdad42b865b6d535a83327b26db657baa6b02d39702fc9b0f52fadb616adcf26e4a5c332e52db731e3b1ae33f73626b498b3b6358baa013692af39367f28522cb4362bae00b48fafa12fbeb67531d2b0da3414b9b0b230b7fcb20f2c4baaa7acf3a8e82a4fb4fc31a4316b2ecc30dd2c5b38a3a59cac7ab24cb448b59fb35ba5212d7023b3b2b63356324fa988b1a0309f2a34b6baa46a2c92b3432a39b6b335593516ac7c2e2fb9f1b466ac92b42a32483819b51ab3212ce43630b5472cca3710b77ab53eaf74b0f332232c97b76930f0b073355fb172b1a1308732aa2f35b79a306cb4062c16b6a22fe6b27cb51b32222c41b1343689af6120992c4db108308fb00bafd43573b429b9e3367339efae24b850ade5ac7eaac3b4f5307c3064b6fb316438e129c6addaa8423166b7a029aeaee737f4a92237b8b40631762d51b30d2a1e3589ad46b78d9caab3c6addfa54db7deb60e3068b07ab9b9326e2c31b297b52533e42d1739452fbd2e08a90133613151353bb07baffcafc7325f36a5af24af2d2d6eac0e34d1b81db3c7b2fcb812b4d8b35cabbb34e0ab75b414a4773429b4aeb3c530f8aba631d2a01c3270b4a89c9cab22b5abb6e43102ae63b443a6d134e1b861b6f2b34136da2c9732e2b4c8b4c82fa823ae304baace2c4b299632f332efb6cfb84a3413355bb4e3b7eb26f733ac3782305eae93b32d1c20ad54b462389e37d434e8b3543379317b2f5035823152326b31cf2d97b75025c2b080b531376a37cd354a2cc1353f1b022baeb4c12edf238bb596a24ea5b237a83343223e2c3d13d8aabcb434a93cb248b548309b2e8ab7e1a85a3493346434c0137bb383b4e12d492e7a35c43407a818b55cadda2bd0ab53b0382c07b057b1f52c58ae862c202c05b537b0203609b595b80cb82824601e6ea91f35a3b93c2ea736e2b452ae23b2a7b31cae43369e372bad0f38601e623769a3cf285f2eca3a3233e8b123347dae29b0a9335620e92c631daa2d3e2f1cb89e312ab423b6baa3a62413b23a33bfb515af08b194a1e62c2db438329e2412b81cb8149cf0aacfa62aa1172c1634d1b417b5b4b4ed2c4c35a2af62b935b728b3ecb01caf2524a1ad7934ddb153b3423093b6662eee27d6a9da365ab165381cac5eb52c2595b46db2ba2ec03052b13934f33424368730d53505b870ad6ab31428e3b0afa90638cea9142ff2ab2e259716ab2ffc340b374c2d0b31ecb403a6053468333d2b953332302235f32c16aebf2d44ba8d2ecdbb5fb7ba2f1630f9b417b2dc3524b0cc383990fb2eea367c2849b482a7dfa0d6b560ac59b2e833bc20d6ac44a7a22b463146b474b4b9340eaf0fb6972fed2f8cb6da2b4a32672e683101b6aa2d2dad21b072335336ac379f3582a7092e8f32252879280b330ea43db217af6c318f12d22dbb3169ae78327b2e48b52b3648b014b778b45eb42423afacc3ad4d31d02d682d032f453182b83caea9b8ef20109579341435f02a47b0263084b02b8e3834c5b0e432cb311bb428341832862785b257b3a4b0a33016b4932f4cb5d7aad4263b39dfb554360629042ffc279e2f1da93236d5afde347fb15fa490b4a02f30364ba597aabcb40aacd4b6c6b3d42ba335043af3b025362c2c18a9692fd824cc27fcaef8b597b68c31caac23b19d14d632792966b13335c2363b3031344b35d634413125317b30c0b4fb31b2b5b8b2993530335236b5aceab4fcabda261226c83436ae1a2e28b4c12fb7946cb2efab16aac8ac8e3733a728ac7fb88db3f5acd5b1df358aa991358222981f49a5ec1b5c37f3b841b893b0431edb31113040ae64345d34a62a2234a62cdcb0d9314927711b2b3628b925b611b52aae17ac82b638b461b34339bc370baf38aeeab414b800a844b4c8a5d8b683b21729432e232fa6b55033d8b26931ba99e3b569ad3f2e3e31d2a88d364f3463ac6a322238313840b5edb35a3686b4a8b527b0302f01a790b1c4b118b286b3e138172d95332fb05aaa1ab2abb9e5b5beb7cfb81cb5aa351da28535062fb232b4b37b303b32eeb0c930d830b3b0cdb410b5ee25972dc7b2481fe1304a31a236763548b433378cb07834aa240eafbbb7e0a9abb623347c36d1b22b2d34b261b19f3249a170371fb04634ad26ce32a7afc6a2eaa726b3a931903114319236323669ae612e55b1651e20b1ee36392e77ab1b35a4b84eb8cf321d244b2cadb53c9e49380d33cc2980afc0ae8db8682c3b33acb68bb1fdb415b450227cb8c731243236ac4bb935b45635e4b9caadf9b47032cfb1472e02319f37cdaa232f55b6443574b2b7b06da396b64b2b6a34a32ce2b6a7a76d320bb8a6b46833e62c692fc8376d2ff631cc39e6acb6b214306d2e88b050b4f838832f52b25c3354a3b8ac1f3185b5412daf378b3062ae7bb0e1aea0ac76aed63369b75f337fae732ed2b4ae3554351a347ab61733ed2eddb6a79e3c2ec130102afb2c4d363e3521ba6eada2a97eb772265735a5b4c12dd5b3bfb15136a4b0e82f00b8f73727313eaaec3007b60f358badfbaa34b0f7b0f131612c9331d09864302aafc438b4b66435d9354ab3c43842342734ac1f6eab2aabae2efe351c2acbb30933dc33a03160329dacf4b367b300b7fa2bcfa584af52ba57b5ca34a8b11c2ef2b52736f2a8022df938733645ad11b191ad503023a7d5ae4236d5a7eb386c26a337292e203433b24eb3e02f3d34f82080b5cfb51ab694b7ba27cf349999d5b88caa8ba97c3836b6c4ae4c37b72fd430b527a22da232efb28033e1b5912d1f2a3cb1a29e4ab194b412aebbb48b2deab4103209b5d8b05db1aa35bc2d20ab373080b8a2a145388928ac249f30a83345b82eb7ca2a1ab296356bad23b86eb5e1b9e3ab34aa39af9c342035aba98db84031c530552d313403b54d381bb46c3238b7f9b37c3664ac2633192c95a7d3350fac9bb7ecb2b430c42eb4b633b2feb044aedcb55cb139b29b32e732fba8b3aed828fdb2e4b40ab46fb1f8b4af294032c9b6be1e78abde2d329c38b666b50b206534bbb8c92d17354da9b630fcb138a8f1b602b90c3436b4c6afebb505b6e4a93eb27c2ecf2f802d0c2cc23690b4b2a3d634633515b646adc4356f30752a71ab8bae052d61acc6a16325cbb191303a3a9ab7892d7eb2f23059228a320435a22ad73404b9cc36642e9f3438b524b30329309dda2ca1348db72d342b18f6aa4bb64d302ab9873161b68d231da801351caaebb155b07235963097ae0bacb6b6a42e86b76e325d281a39b3b412a7b0371835f3b1c82b1611c62cf3b17da6d1b6a028ab2f00ace4b63b343c2e35b2cdac6bb663b499b0413578b17a319bb03e2cd6a8773393b0b9b6e9b58db057b5052ff8af14396eb7f82ad3b439b0d628f9b80d2c1238c1af712c62349da9f8b0deb87b31bfa755b45528feb085b03031b9b57a34fab5abb11d38d234ac2605b487b0032521213aa639a054ab2bb66ab0d233dab57baa02b39b27a6ae513655309cb5bd31a0b2efb05e2d34b0a333a0b24bb19db1d6b0b33461315037d93868b4d42408b5a7b264b185b47fb4f0a834b53333d9b3023ab8afeb248ab54c37c934142dc2b5352bccb44136ec32882636b934b4e2313035be34cfb6292d583417b75198af2c25adecb792ad99ad12a9d1b095af71a1912f34ad03af51261bb8143069896c211c342db838b8af2cd297d429e8afaf31fa33243076ac82b894323935833186b3c3315c33aa367eb3beab6432e123c7b4172ff3300126822b6cac5236b4b306b1111a853367b69ab55bb87d20c8b494aea8317db1e92b1eadc0311030ebb1c5a97734bc2ef9b59b31a3b40a366134f62c60292aabce3158ae072c7533ceb045b11bb58db4d2b73e32bc371836f4b633321536d0aff1264bb8912c8cb4d22aadb6a4306234b3b2ddb49cb88eacf9b0a2b6c02fb3af9ab4beb09534283173a84c31f7b46c200a2905b0732fadb25c30b8303dba173584ba45b163346ead45b6adb464341cb8d43491a319b5fdb059b4683343325ab7e6ae62b38835c6b5dc368fa0f8b4cbb1dbb273b57d2640b5a9b87eaeb6b76fb2c6b373b34738b723332c99b798b15d2b61b4c43325b5e6ac04b19f297ab1a2ac0aac48b22630142c11a65dafd03209ac3fad2a274137bbb596320cb86e2bbc362e30d629cbb0432050acb9b2f5acc82a90b2e5b2d1b5af2eaa30ec2b21340726b42e322d43a68ab0bc38baafb2b16c2f4f2c02b6cd346fa6c0b415355a3168b59937a2318d99cd367229b89fa6b1fa3013300b1cc2b27228c0a4b4b54bb6d53090b9062b16b0983538b0a431caaec6274d32eaaf433440b4092de333a631193888b2e6b540b4a6314230ff26cdb48f323db5f5341c3a452efd3037b0beb728b1acb430b65d37ab2c7e2e49adeaa920346fa08a30b5aadbad04b63bb879acb92380b5b6a055239cb8d3b3f82e9f2eefb46d295f34a53285a601b062b10030ae2a6231832800337826b6327233e2b2bbb79a36642c55b71bb29a32c8ac62b0fe357ab4d4b2392baf2e3f30cb33cb2a2f306fac4f245a275e2b9b32bf301122eab29c24b09f9c2ab8387bb0392f9234b626003978ae8b2dd5b203ae422c3d33762d62357fa6afadc839bf2e27b2f0b108b401343bae5a2859a442323eb399b4ad2b2431b02decb2cb315db276b00ab490356b312d315535033a4fb751b245b147b39cb420ad34b5bb2d14231d305830283163ae8938022cf7b749b471b7c41c28a802b04d38b2303632f6b3",
    "b1": "a997248f5f0269978117e7153217989bed153f92fd9a4e14c2970b0902988195ed19f79c691323986597c818ae03779b1f9472097090e29e47833b96cb97e714aa960a18c6172a1cef14e981a40a3b09fc18c298e5102f108e95529a3a12588be10cf88d1c0e60107f10a690b19d0211028c7993dc9b108820902a100485d711",
    "W2": "a7b17d20392839b529a81119f2a8a3b017aee6a86424e92e0e32c3a9eeacd32c252ff8ac593092314833fb34e6b0640f1c1c3fa45baf35b4232256b03b2480b039b26529f926972e2e36462a772d24b23bb3733434b1e0b105a020a113b130abfb329ab0d5b35b318a2d27af78b443b0df9fa132d2b1dda37e2fbbb6dd31abb2842b9f292e323a2ebab670b0452662a84e2d6a30e42edfa83bb51eadce2c20b2022f2431432556b838b35b3532ae9bb5f732a7162e324b2a152d1a2cce2f68367e3170b4412d7b32ca32d4ac95323e31443777b0b0adceaaaeade1b0c2b7f6ac8e2cb5ad5935ec1a48339029af30f429c93346219f2d192c4eab9d2c5e36dba9bd2c35338a32a71ac0b6d72d49b62634e727663112b160b4a2294a1555ad8e220038aa30ad31d6302a34f3a8ceb5c6a8af306fb168a7e4af26344430ab2f532fd79e9f28a9b227b69326542bd03267347b34ddb1c9ab942d213757347932b2acdab0643040b229a986a56e3402acf5b6ec2dcbaff32009a4efaa7b304fb4aeadbd2fe5ae90b49d2f57321ea08cb50230b3b4252a05b18eb225ad44a811af902e88ac7431a729d7afecb0b9b4a22a1bacb5b614ae1ab17334e0b413b47332b43337b03cafb02e97b644b55aaa6334e1ac78246432f3358bae1a9a0f354db4c931b1b1a12426a96ab191b1062a9e319da58721662e83b13430c3305b3730ac4fad5a309c3006ae9dac272eaeb09bac6d248b31adae39b572b51cb399b33ca79db236b05bb0f5ac6a33912fcb1e38adb5ac142b782e45b14e1fb9b5bb283b3002ab2335e9b02b29629bc9a42da59424b630e1334fa0aa30cdaf4130eeaf54322a3676a43bb63bae06332b2eeb28b6314d2d0433443154340b2cf130e7aab9aed634dd28cf30d42765ae59000f342e2edeb4aea8a4346ea808b53c357e30088f9fac0e3398aeac317a2e162977218ca78831a5b087b4472ff32e1f2cc12c57a9d3300ea0d8ad0dacb335e7b4aa2dbdb463a401add5b53d2c6eacefb522300b3439345d2b8220cc33b9346f3257b170284d2d31b5092d5c36652f4c34e62779a29d2b9eb069b13f33d733a5b17a311ea4622b4b2d473130240a34822aae2f7bb00636d12dba2facb495b0f532c22969b5b531f9ad033436a37e28533211b17ab1c7a7ad307fb3b2b4b230d9ae71ac42b5c1af1a86193155b0b634d8af56af64347328d6ada2a780332c34a6b041a12924aeb1fe9f24198632cd3211acc9a8ba2f5eaeaa2effa374b16db1193412a7a534cbad622acb36fcb15fb54c33aeb0539b7cae43295eb1d3a7022dadb1b4afdaab04b4e0ac1e25adac1eaabd9970af9d2f013440ad442d30aa162c6e34c22d02b5352df132a5a89431deaf31b49022fcb45ea92a3364b50ab3dfb18da120ace5337e31612de725532cd6b00d3153b2ba309032c12854a7b52c08b8bb2a27a51ca1da2f69a1bcb2f4a551239224202905324ab27029353303b56a28bbae95ac1ab18da8483510b37cb267b3c2342cad400c24ac58b155a9ed36e133d5b0532fb4b005b5c5317f32e9314fada8a72e307e2c7eac87b4fe241632c2b2332f422b3d30ea30e2af94b405ae083116aedcaf1fb5f634ac30453056317c2e85a965a7d0ac78373b2e27ad352dbfb14434f1af922d0932342ca424841e63a183ae17ad2c346ca486b0f1ad2b32b52fa52ee9a9772e7fafd9a9333175309ab1d6b0f924e7345db5f029a3b371af4facb9ac08b0a38e06ae40b47930bc30b03059b4882a842f99b1359f15a71c35813579b4033722b126317e31aeb1832d3ba54e2e3a29c932242304a85232852e1f3078af0bb27caee1ae3a308c3113a60634c633ec29a7b07e34b0a24e25f3ab05b11eb4212f84b04b3824b234b3822d4d3407227b2e35b59faf1bb24eb5fe245ab676b39caebbb08b2c532a13316f306b32c80c072bc22d0a27a9acc6abaab0d32c3da28caf91aff8b289311924d1b0a53065aef5b3c03133ae7fac898081324e1ddab2bb2e43ad4735561e58b6a72b883570341835c4b10c31f5af492922b353affeaf19319d30a32aa4b3c0aa48b0f730a432f93148b0bcb701b5c6338aac3fb09e342233c91fe0aa3ca207297134e12c113063329bb238309eaa1fb704b6c3b1052e3725df2884352eb3309c0d2b28b0191d86b4fa33baafbe356d9ed331bca0949d71ae84253f2b7eb0b0a5eaae803607a4623468a8df2978b30eadaf287c34ef28d4332d2f963457a1c53686ad57b0d8b29836a33087ab0b33c42f903563b40aaa12ad98b5013292aa3a30b93608acbeac872593314bae26affb312a2cc7b5e4b23e34692c76b0b4a7ea152130b3b3c1b02f2a79a4842efdb21c2c2a32752d6126fca4bfb0e3a10a331fa500b19bad972824ac762dfb2dfd300334943145b110b1fea22cad82b202ad2bad62afccaaa0327f28f836f9262aaa5ca2182a222dd03552b01131c72735b43d30e432bbb0f29b06b51cb3b031eaac2cb2cc29ffad02ae38305cb2e3b0f9267c31033384af4d3170b02b3569b282337eafb235af2ec4b1f4ad7130f431cea1201f3b30f22552b39cb05829e03292346730752a3b30a3aff0a6a0b00fb38c2b26288b2f392e70294b2e769fbb2c21b0ab2b842f95afec1f6eafc9ad66b26033fa31f72d963560a8413823a9be299aac202cb7301ba509355f29bf2a2fb1bc2cefa5d5ae62b08e37f7a91fb4dcaf07a1fd2d452e9f2f68af812be7b004af18b1c4b2e431ca2f8431e81c6630fdb4e430d827012f4f2cafb13cb4e1af5e2ece2fbdb384b112b010acb4b306b4adb0c523c5b21ca5043268b35ea8e431ceac81ae843229ab4132d82cd5ab6c2d1f2bca3135ae661e94a1b62fceb1c031472d73b3e1325736af310834b723b4b437b5c929edae29b1a9a1e72e10b402ad4334f5b1ed30b6ad392e3132213024318eb25632ceb0e329d329552dab2bbab099b3f0306e27503280a0f1b378312aace82b65afb63401b1c9ab102584b22f2b9134ebad932cabb732a002ab99b01632f2aa6ea86bac76b55bb2a831fcb55fb0ddacee345d303bae6a3335b0b531b3b3093293acf6b1d1a8662aaead962e68aeed2e0b2de4a113ad0427332ecf354e3179316038dc2d752df72fd2b106adbdad4e2feda72f274732f630a4a796af62324d3080b467ac082881300531182c0e2faf1d97ae70aa05a01d995fb510ad642154abe5a8b82df5b1a1b4afac5c30e534af3027b443a966290228162e30b1f921cb2ca730a5b0ddaba6b711af23aeb8b15427989d65b6e0283cb17a2d54b35e1d60aab72a7925c6a958b425b49eae660ff7ae142f8baa8bb16daa27295c2b523587af1fa3052d259db1ad60b42bae1ab2dfb4a4b0e6b1df2ebb2e18b1bfa942b570ad632c9e3011b0f9313bb413b3f32faa34962d35a03eb6b9b5e0250cac40b25c22f2245a310226ec32941577a6023285b1ee37472c6b2ce42d503579308bb4fc33e9ab79244cb3002e38af6f2464aa66359c351e1e471a3bac3e3272abbcac9a3133ae24a280b4eb2f7935492f2f34cdb0462a5db108a960306b2415b474b3c9b34e2dca35a63358b51b34453149b463ab3132a92e46b3132f11b611a9653288198dad8a3094317233d33083325a307929a22893a94db34bb1adaa27281734b2a2d9313432dca8aa2ea2aa8733ae2f91b24f27ab344da8172d292edeb1c82ddfb318affe31062ce4aa052d3fa2301b662ff0afe01e42ab7ab438b166345c2c81b35eb429a56c343ba86b33f72b792d91b34f316fac4030a4b00028fc2b8ba6d12c67aa32af7734da314cb00b2c32b0bab2bc31d1b4c33105aac2a8682c752e6bb0702f23224bb3ada62f34b23430302db38c2dd430203074301a2c90a2e5ae61b5c42b0428562e8cb3462fdb35882318a23c2bbc35d2b1ac2889b03fae6ea095331c351d2d34ac0c37f2ac73ae02b5f2337a31d536d5b31f3460229030a9b4243473304c9e4f2f50b090ae1dad3ab5e130a632fbaf032c0eaa762b0baffeb0f131aea1b83506203eb1e532222f1330b73266af21a43da63fb316aa8d334029a3b363b471340da333afbbaae5b32634a8ac3534a31d88333c2ac130912f252fef2a9ca8783503b1f59523a1a92c48ac122ca2a77eb47e933ab2e8b433268a2dff2d602e78b5b8b39335ecb1992ec9af27232aaf5e2eba2615adf1b1e929deb2e52c673799330a307ca371b57eb349338bb0e7aff3a703b2fab24ea00ab12cb05cb0b42daba833a89b2e74a690ab452dcf2fa0a89c235fb0f6b06730522d212df72a121dab2f163134313ea4d233b928ac2e5d2a0d2a13b00e3004383facb52134ad4fad38347faf3ead8ab4c2afc4b4c52e0c3847acc8b42e312cb0173354aa72af91a66230b5a8a7ad67a92ab3c62cfeb403adff2f8d288d2ed4a19f3248313f2d2a31613037ada8b3d72c4a2c1c3254b142b14d2d611b5bb5442c47b56ab1dd33033512b3f1ab6a215235bdad54ad33343eac173031b4b7a7b2b4f9b1bf308ca032ab4db0fda8d92a8c316db7abb0052756a3822ce629d7aafbadb2a9813229b0b9a12c286c34d4aa6731f832ffa0a6a876ab68a9ed307c2939295eb281b1093681a7a1b01234952a672e45a3bfaa4ea887b36ba4022a45350d23b1b626a4ad37093185b0df2e39b0dbb3c7ab1a253ab57fb25b970bad742ce1b20b30aeb2ad34e1b439b2982caf34e72a5a2d7d1cd2b186adebb229b37b2c5ab2e22a63b30b2bc5add533cf2a0b3330b079206cb108adeea920b0f3b2f9b39fb6fab11b2f86b4bcb30d353aaf462b382f422f4ab46aa992b4e2367db4efb1e4b140b4a733dc366dad80b357b67aa92fab66af3d29632c562b91ab4cb2652f951a08318330f7b322b403b009b22d301a2cd231472c8e2942328b2daa241f34db25cba9c2b03430cd32c2a228338bb58c9b44a056b09036efa1d9ace191adb54bb0913115afa5b05d33b433bb2d43ad6da9feb0a9b1e2b160afd8b153b4b9286529d3349baf362ba02a2eaa16307130062cefa90facf0b38fb006af54b2fd2cc53223b16a2addb1f8aad0afd22eabafb8314da94eaf11b32e291331602a52360d34ceaa32ab5c2dbfb09632fd2bd429bb2fdf343127e1a545346e31c2ab7b086f28192cabb35bb409b168b154af272faea7c427f7b243ad732ec82ed6af2d273936c09be0af42a7b7231da7ac9e12b44eb1e2a6aea7b3aef2a728837a30ffa48b2c1cae912e792cd03054305a2e4ca5b529d7308faf12af0c25d42c0cb04ea4e9a343b642a9a5a4e0a8c0331cb385b01db7a3ad81acbaae4432df30ec222eaccc31349ba1b0d8b408ad2330703686215fb0bbb2582effb4892e9a30ae2ce6a85c2e022767b03735f7ab6030d8b638331f348e1d882e5a2e46af412e89b0d8ae5e32833102b02bb042af1330a4b1122c343010a2bf3107ab89a88ab1b33150b0dd2b4ca3f4b1f9aede33aa28aeb1e02a28a7e338912a38ac043430b57724a6354aa5e7b47fac1c34492cc9b0b9af52ae0b2cb433f631162d89b382b3182e9bb320a9b82a6eb2cab1d5a3be2d0232dd2e652237b013b2e2ada42c9c2bf0b570b0d0322f2302b4ab36f89d632c7e2c36ac26b0f035b631aa30552eb8b13433ceb07a2aa0b4c5b56c2b9ab0a6b5f02d0236cd33e0a98616ed305932b4ab361fdb2a422e08a67ca5c433e032522ddf3452ac95346a2ea338cab2b7afaa2c722c863594af19319732e12f66b1fe330033be2e43282629c5a25e2f",
    "b2": "88800d8d4898f11380913380b6898e88cf92559f4981a20c6f98ee8de31a1d97d71c2892a5983f1356903e9cd08e6796a2912b947a9586066f99b389b11bc597",
    "W3": "9a1fcdb283b30cb2032a6d20e229332fbcb0143502ac05afb92b0fb7ecb2313341b77334913536ae8e2c3f343baf703227b118aeb7b1f32d6f34aba835b57bb3",
    "b3": "ac90",
}

_W_SHAPES = {"sc_center": [20], "sc_scale": [20], "emb": [50, 8], "W1": [28, 64], "b1": [64], "W2": [64, 32], "b2": [32], "W3": [32, 1], "b3": [1]}

_EPS = 1e-9

def _load_weights():
    out = {}
    for name, hexstr in _W_HEX.items():
        shape = _W_SHAPES[name]
        arr = np.frombuffer(bytes.fromhex(hexstr), dtype=np.float16).astype(np.float32)
        out[name] = arr.reshape(shape)
    return out

# ── Product universe ──────────────────────────────────────────────────────────
_CATS = {
    "GALAXY_SOUNDS_RECORDERS":     ["GALAXY_SOUNDS_DARK_MATTER","GALAXY_SOUNDS_BLACK_HOLES","GALAXY_SOUNDS_PLANETARY_RINGS","GALAXY_SOUNDS_SOLAR_WINDS","GALAXY_SOUNDS_SOLAR_FLAMES"],
    "VERTICAL_SLEEPING_PODS":      ["SLEEP_POD_SUEDE","SLEEP_POD_LAMB_WOOL","SLEEP_POD_POLYESTER","SLEEP_POD_NYLON","SLEEP_POD_COTTON"],
    "ORGANIC_MICROCHIPS":          ["MICROCHIP_CIRCLE","MICROCHIP_OVAL","MICROCHIP_SQUARE","MICROCHIP_RECTANGLE","MICROCHIP_TRIANGLE"],
    "PURIFICATION_PEBBLES":        ["PEBBLES_XS","PEBBLES_S","PEBBLES_M","PEBBLES_L","PEBBLES_XL"],
    "DOMESTIC_ROBOTS":             ["ROBOT_VACUUMING","ROBOT_MOPPING","ROBOT_DISHES","ROBOT_LAUNDRY","ROBOT_IRONING"],
    "UV_VISORS":                   ["UV_VISOR_YELLOW","UV_VISOR_AMBER","UV_VISOR_ORANGE","UV_VISOR_RED","UV_VISOR_MAGENTA"],
    "INSTANT_TRANSLATORS":         ["TRANSLATOR_SPACE_GRAY","TRANSLATOR_ASTRO_BLACK","TRANSLATOR_ECLIPSE_CHARCOAL","TRANSLATOR_GRAPHITE_MIST","TRANSLATOR_VOID_BLUE"],
    "CONSTRUCTION_PANELS":         ["PANEL_1X2","PANEL_2X2","PANEL_1X4","PANEL_2X4","PANEL_4X4"],
    "LIQUID_BREATH_OXYGEN_SHAKES": ["OXYGEN_SHAKE_MORNING_BREATH","OXYGEN_SHAKE_EVENING_BREATH","OXYGEN_SHAKE_MINT","OXYGEN_SHAKE_CHOCOLATE","OXYGEN_SHAKE_GARLIC"],
    "PROTEIN_SNACK_PACKS":         ["SNACKPACK_CHOCOLATE","SNACKPACK_VANILLA","SNACKPACK_PISTACHIO","SNACKPACK_STRAWBERRY","SNACKPACK_RASPBERRY"],
}
_PRODUCTS  = [p for ps in _CATS.values() for p in ps]
_PROD2ID   = {p: i for i, p in enumerate(_PRODUCTS)}
_PROD2ORD  = {p: float(i) for ps in _CATS.values() for i, p in enumerate(ps)}
_LIMIT     = 10
_FLAT_THRESH  = 7
_QUOTE_QTY    = 4
_FLAT_QTY     = 10

# ── Forward pass (pure numpy, no torch) ──────────────────────────────────────
def _silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

def _forward(W, x, pid):
    h = np.concatenate([x, W["emb"][pid]])
    h = _silu(h @ W["W1"] + W["b1"])
    h = _silu(h @ W["W2"] + W["b2"])
    return float((h @ W["W3"] + W["b3"])[0])

def _scale(W, x):
    return np.clip((x - W["sc_center"]) / (W["sc_scale"] + _EPS), -10.0, 10.0)

# ── Book helpers ──────────────────────────────────────────────────────────────
def _book_stats(od):
    if not od.buy_orders or not od.sell_orders:
        return None
    bb = max(od.buy_orders)
    ba = min(od.sell_orders)
    bv = float(od.buy_orders[bb])
    av = float(abs(od.sell_orders[ba]))
    mid    = (bb + ba) / 2.0
    spread = float(ba - bb)
    mp     = (ba * bv + bb * av) / (bv + av + _EPS)
    imb    = (bv - av) / (bv + av + _EPS)
    pres   = float(np.tanh(imb * np.log1p(bv + av)))
    return bb, ba, bv, av, mid, spread, mp, imb, pres

# ── Rolling state ─────────────────────────────────────────────────────────────
def _init_buf(mid):
    return {"mids": [], "imbs": [], "e3": mid, "e21": mid, "e8": mid, "e55": mid}

def _update_buf(buf, mid, imb):
    buf["mids"].append(mid);  buf["mids"] = buf["mids"][-60:]
    buf["imbs"].append(imb);  buf["imbs"] = buf["imbs"][-5:]
    buf["e3"]  = buf["e3"]  * (1 - 2/4)   + mid * (2/4)
    buf["e21"] = buf["e21"] * (1 - 2/22)  + mid * (2/22)
    buf["e8"]  = buf["e8"]  * (1 - 2/9)   + mid * (2/9)
    buf["e55"] = buf["e55"] * (1 - 2/56)  + mid * (2/56)

def _rolling_feats(buf, mid, imb):
    mids = buf["mids"]
    ret1 = mid - mids[-1] if len(mids) >= 1 else 0.0
    ret3 = mid - mids[-3] if len(mids) >= 3 else 0.0
    i1   = buf["imbs"][-1] if len(buf["imbs"]) >= 1 else 0.0
    i3   = buf["imbs"][-3] if len(buf["imbs"]) >= 3 else 0.0
    arr  = mids[-21:] + [mid]
    z5 = z21 = 0.0
    if len(arr) >= 5:
        sl5 = arr[-5:]; m5 = sum(sl5)/5; s5 = (sum((v-m5)**2 for v in sl5)/5)**0.5 + _EPS
        z5 = (mid - m5) / s5
    if len(arr) >= 21:
        m21 = sum(arr)/len(arr); s21 = (sum((v-m21)**2 for v in arr)/len(arr))**0.5 + _EPS
        z21 = (mid - m21) / s21
    return ret1, ret3, i1, i3, z5, z21, buf["e3"]-buf["e21"], buf["e8"]-buf["e55"]

# ── Order logic ───────────────────────────────────────────────────────────────
def _make_orders(sym, od, signal, pos, bb, ba, bv, av, mid, spread):
    result  = []
    buy_cap = _LIMIT - pos
    sell_cap= _LIMIT + pos

    # Convert fractional return signal → price-unit skew, cap at ±2 ticks
    skew = int(max(-2, min(2, round(signal * mid * 0.3))))
    half = max(1, int(spread // 2))

    # Passive quotes — hard-clamped so we are always makers
    my_bid = min(int(round(mid - half + skew)), bb)
    my_ask = max(int(round(mid + half + skew)), ba)

    # Active flattening when inventory is large
    if pos >= _FLAT_THRESH:
        if sell_cap > 0:
            result.append(Order(sym, ba, -min(sell_cap, _FLAT_QTY)))
        return result
    if pos <= -_FLAT_THRESH:
        if buy_cap > 0:
            result.append(Order(sym, bb, min(buy_cap, _FLAT_QTY)))
        return result

    # Normal passive quoting
    if buy_cap > 0:
        result.append(Order(sym, my_bid, min(buy_cap, _QUOTE_QTY)))
    if sell_cap > 0:
        result.append(Order(sym, my_ask, -min(sell_cap, _QUOTE_QTY)))
    return result

# ── Trader ────────────────────────────────────────────────────────────────────
class Trader:

    def __init__(self):
        self._W = None

    def run(self, state):
        if self._W is None:
            self._W = _load_weights()
        W = self._W

        try:
            bufs = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            bufs = {}

        # Pass 1: cross-sectional stats
        cs_mids = {}; cs_imbs = {}; cs_pres = {}
        for sym, od in state.order_depths.items():
            st = _book_stats(od)
            if st is None:
                continue
            _, _, _, _, mid, _, _, imb, pres = st
            cs_mids[sym] = mid; cs_imbs[sym] = imb; cs_pres[sym] = pres

        if cs_mids:
            vals = list(cs_mids.values())
            cs_mid_mean = sum(vals) / len(vals)
            vals = list(cs_imbs.values())
            cs_imb_mean = sum(vals) / len(vals)
        else:
            cs_mid_mean = cs_imb_mean = 0.0

        pres_sorted = sorted(cs_pres.values())
        n_pres = max(1, len(pres_sorted))
        ts = state.timestamp
        tod_sin = float(np.sin(2.0 * np.pi * ts / 1000.0))
        tod_cos = float(np.cos(2.0 * np.pi * ts / 1000.0))

        # Pass 2: per-product predict + order
        result_orders = {}
        for sym, od in state.order_depths.items():
            if sym not in _PROD2ID:
                continue
            st = _book_stats(od)
            if st is None:
                continue

            bb, ba, bv, av, mid, spread, mp, imb, pres = st
            pid = _PROD2ID[sym]
            buf = bufs.get(sym) or _init_buf(mid)

            ret1, ret3, i1, i3, z5, z21, ex_3_21, ex_8_55 = _rolling_feats(buf, mid, imb)

            n_below = sum(1 for v in pres_sorted if v < pres)
            cs_pr_rnk = (n_below + 0.5) / n_pres

            x = np.array([
                spread, imb, mp - mid, pres,
                z5, z21, ret1, ret3, i1, i3,
                ex_3_21, ex_8_55,
                spread * imb, pres * z21,
                mid - cs_mid_mean, imb - cs_imb_mean, cs_pr_rnk,
                _PROD2ORD.get(sym, 2.0) - 2.0, tod_sin, tod_cos,
            ], dtype=np.float32)

            x_sc   = _scale(W, x)
            signal = _forward(W, x_sc, pid)
            pos    = state.position.get(sym, 0)
            orders = _make_orders(sym, od, signal, pos, bb, ba, bv, av, mid, spread)
            if orders:
                result_orders[sym] = orders

            _update_buf(buf, mid, imb)
            bufs[sym] = buf

        return result_orders, 0, json.dumps(bufs, separators=(",", ":"))
