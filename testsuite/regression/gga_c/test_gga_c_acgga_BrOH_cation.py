
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_acgga_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_acgga", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.684341861139999e-02, -5.684424060589999e-02, -5.684658541991726e-02, -5.683438374186855e-02, -5.684096297367738e-02, -5.684096297367738e-02, -4.480714261216336e-02, -4.481211239763611e-02, -4.492726570120296e-02, -4.472777229944830e-02, -4.481315252584477e-02, -4.481315252584477e-02, -2.909089742838183e-02, -2.886273178188619e-02, -2.336959670386420e-02, -2.363170123088305e-02, -2.365901649656633e-02, -2.365901649656633e-02, -6.856641964922344e-03, -7.447527749193706e-03, -3.171398407057318e-02, -2.307260850890586e-03, -2.469401310577518e-03, -2.469401310577508e-03, -8.866687741665435e-09, -1.174969872005244e-08, -1.156412328700597e-05, -6.688279266357788e-10, -1.696796440766452e-09, -1.696796440766452e-09, -6.190324842948726e-02, -6.210409921196988e-02, -6.191152711995918e-02, -6.208883595589141e-02, -6.200543475921054e-02, -6.200543475921054e-02, -2.368226390308065e-02, -2.415649217069486e-02, -2.263428309333321e-02, -2.304195973421296e-02, -2.446968466976573e-02, -2.446968466976573e-02, -3.987546970454668e-02, -5.628404283796622e-02, -3.720877062274228e-02, -5.168890814048208e-02, -4.161362881456544e-02, -4.161362881456540e-02, -4.894006736574086e-04, -3.647740722995930e-03, -3.822330755657612e-04, -7.275179169972620e-02, -1.236507101053691e-03, -1.236507101053691e-03, -2.634294028573680e-10, -7.049099355442567e-10, -1.186519815342055e-09, -1.023469141376160e-04, -1.188343938442887e-09, -1.188343936274483e-09, -6.054664393334706e-02, -5.530543755917436e-02, -5.704542589383118e-02, -5.856655057216256e-02, -5.779634255599836e-02, -5.779634255599836e-02, -6.166317490858522e-02, -2.846332428786167e-02, -3.566710477572831e-02, -4.448260201276354e-02, -3.985222934138795e-02, -3.985222934138795e-02, -5.623524118813103e-02, -6.395352277547291e-03, -1.087733666770852e-02, -2.448626182602434e-02, -1.689387325110307e-02, -1.689387325110308e-02, -2.766589851527066e-02, -8.829946410989636e-06, -3.264634752833673e-05, -2.923291062030520e-02, -3.222535986302164e-04, -3.222535986302164e-04, -3.386863033430498e-08, -6.853978322396426e-12, -1.109406949986735e-10, -2.486514522355987e-04, -1.064933621754832e-09, -1.064933617851704e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_acgga_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_acgga", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.308335243447729e-01, -1.308336795689088e-01, -1.308343956060468e-01, -1.308346573578884e-01, -1.308373198330669e-01, -1.308370304358655e-01, -1.308229917073110e-01, -1.308239097788487e-01, -1.308300815024659e-01, -1.308316177901250e-01, -1.308300815024659e-01, -1.308316177901250e-01, -1.104292654431753e-01, -1.104307698001983e-01, -1.104346101017667e-01, -1.104363280723996e-01, -1.105628738483250e-01, -1.105605991197703e-01, -1.103447031977220e-01, -1.103431580328321e-01, -1.104573921147878e-01, -1.104161409991033e-01, -1.104573921147878e-01, -1.104161409991033e-01, -7.894734024909988e-02, -7.862151546719250e-02, -7.872439268825883e-02, -7.832643346882419e-02, -7.091949256555209e-02, -7.140550214665091e-02, -7.169633276200431e-02, -7.154293597165086e-02, -6.943724295596700e-02, -7.408253962660916e-02, -6.943724295596700e-02, -7.408253962660916e-02, -2.955812706203972e-02, -2.853892087017347e-02, -3.151590314878624e-02, -3.027384275376559e-02, -8.473962268855777e-02, -8.183949893976104e-02, -1.179446656810075e-02, -1.165473422251645e-02, -1.102990378098222e-02, -1.914422625467331e-02, -1.102990378098226e-02, -1.914422625467331e-02, -5.725485938709163e-08, -5.508861656048076e-08, -7.604332041715971e-08, -7.277406980561412e-08, -7.238682980774327e-05, -6.936600518193351e-05, -4.241980489500313e-09, -4.285111553324955e-09, -1.017282926261487e-08, -1.394897988180608e-08, -1.017282926304855e-08, -1.394897988050503e-08, -1.295199834970363e-01, -1.295537533843117e-01, -1.296860925787782e-01, -1.297208451657249e-01, -1.295265619928651e-01, -1.295609641072704e-01, -1.296738648629342e-01, -1.297078286777816e-01, -1.296044944241161e-01, -1.296389295840857e-01, -1.296044944241161e-01, -1.296389295840857e-01, -7.734909684503743e-02, -7.735254588585747e-02, -7.826880146005732e-02, -7.828767037739277e-02, -7.533841075118665e-02, -7.522467608414372e-02, -7.616917576351927e-02, -7.604901485589027e-02, -7.871139926854745e-02, -7.901886286513089e-02, -7.871139926854745e-02, -7.901886286513089e-02, -8.351351574105066e-02, -8.378111626749003e-02, -8.298701985542963e-02, -8.293101567049303e-02, -8.296335771271499e-02, -7.949007610378055e-02, -8.279035124186948e-02, -7.898688214757031e-02, -8.176618824881822e-02, -8.706964199980563e-02, -8.176618824881819e-02, -8.706964199980559e-02, -2.789433743172241e-03, -2.767251659474374e-03, -1.760880494842543e-02, -1.753193019496837e-02, -2.243365343024952e-03, -2.147171182004112e-03, -1.168359868654156e-01, -1.169070453679435e-01, -6.826247484440810e-03, -6.497884307696647e-03, -6.826247484440810e-03, -6.497884307696647e-03, -1.703492262502588e-09, -1.665315683200245e-09, -4.512769726343423e-09, -4.472221073366443e-09, -7.685488028200085e-09, -7.430936865499657e-09, -6.104254767760680e-04, -6.073748027634366e-04, -8.597021820290745e-09, -7.152110824751572e-09, -8.597021817065243e-09, -7.152110822393433e-09, -7.607965942815961e-02, -7.559403426501146e-02, -8.039983402373339e-02, -7.992872054798504e-02, -7.916974740070087e-02, -7.869191615722188e-02, -7.792879622150803e-02, -7.744971912194938e-02, -7.857641573851254e-02, -7.809799038574186e-02, -7.857641573851254e-02, -7.809799038574186e-02, -7.337879495953521e-02, -7.296263972006979e-02, -7.449103715160976e-02, -7.414327560834814e-02, -7.979682522600780e-02, -7.938341363075366e-02, -8.190509364914574e-02, -8.153009012589946e-02, -8.133035434973221e-02, -8.095937022140066e-02, -8.133035434973221e-02, -8.095937022140066e-02, -8.498118861925337e-02, -8.477907760735065e-02, -2.806840974173088e-02, -2.792283128314849e-02, -4.177600533593824e-02, -4.138576809483562e-02, -6.615933309451592e-02, -6.576511934829687e-02, -5.482659351463472e-02, -5.484595076345563e-02, -5.482659351463474e-02, -5.484595076345565e-02, -7.274408235600138e-02, -7.220439856521391e-02, -5.428005579549083e-05, -5.399253645721546e-05, -2.003630713277416e-04, -1.951406588305498e-04, -6.910128469215532e-02, -6.790698953226425e-02, -1.904885842864358e-03, -1.820540541717692e-03, -1.904885842864362e-03, -1.820540541717699e-03, -2.158887447837602e-07, -2.108697834061829e-07, -4.417796763593973e-11, -4.412257206066006e-11, -7.239623428663853e-10, -6.988075516389550e-10, -1.457110189557183e-03, -1.437642562862911e-03, -7.580405605238610e-09, -6.428597729816218e-09, -7.580405601701401e-09, -6.428597728664253e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_acgga_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_acgga", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.859065356981574e-10, 3.718130713963148e-10, 1.859065356981574e-10, 1.859096985753588e-10, 3.718193971507175e-10, 1.859096985753588e-10, 1.859137612122552e-10, 3.718275224245103e-10, 1.859137612122552e-10, 1.858669198545945e-10, 3.717338397091891e-10, 1.858669198545945e-10, 1.858928816798203e-10, 3.717857633596406e-10, 1.858928816798203e-10, 1.858928816798203e-10, 3.717857633596406e-10, 1.858928816798203e-10, 1.037613449185223e-06, 2.075226898370445e-06, 1.037613449185223e-06, 1.037819061638293e-06, 2.075638123276586e-06, 1.037819061638293e-06, 1.042464114536882e-06, 2.084928229073765e-06, 1.042464114536882e-06, 1.033447741888961e-06, 2.066895483777922e-06, 1.033447741888961e-06, 1.037737441841090e-06, 2.075474883682180e-06, 1.037737441841090e-06, 1.037737441841090e-06, 2.075474883682180e-06, 1.037737441841090e-06, 1.733309524103645e-03, 3.466619048207290e-03, 1.733309524103645e-03, 1.721095251372987e-03, 3.442190502745974e-03, 1.721095251372987e-03, 1.412688543279379e-03, 2.825377086558758e-03, 1.412688543279379e-03, 1.379700781046154e-03, 2.759401562092309e-03, 1.379700781046154e-03, 1.399600302536842e-03, 2.799200605073685e-03, 1.399600302536842e-03, 1.399600302536842e-03, 2.799200605073685e-03, 1.399600302536842e-03, 1.620324092875536e-01, 3.240648185751072e-01, 1.620324092875536e-01, 1.699243099704153e-01, 3.398486199408306e-01, 1.699243099704153e-01, 9.348009159119282e-04, 1.869601831823856e-03, 9.348009159119282e-04, 1.263926386386177e-01, 2.527852772772355e-01, 1.263926386386177e-01, 1.036556904687281e-01, 2.073113809374562e-01, 1.036556904687281e-01, 1.036556904687284e-01, 2.073113809374569e-01, 1.036556904687284e-01, 6.679813686774626e-03, 1.335962737354925e-02, 6.679813686774626e-03, 7.799192208536592e-03, 1.559838441707318e-02, 7.799192208536592e-03, 3.465056553073542e-02, 6.930113106147083e-02, 3.465056553073542e-02, 2.504236150644069e-03, 5.008472301288138e-03, 2.504236150644069e-03, 3.603049413126920e-03, 7.206098826253840e-03, 3.603049413126920e-03, 3.603049414659941e-03, 7.206098829319883e-03, 3.603049414659941e-03, 2.687176345870060e-07, 5.374352691740121e-07, 2.687176345870060e-07, 2.704636598034391e-07, 5.409273196068782e-07, 2.704636598034391e-07, 2.687872956339284e-07, 5.375745912678568e-07, 2.687872956339284e-07, 2.703285155746530e-07, 5.406570311493060e-07, 2.703285155746530e-07, 2.696057365855835e-07, 5.392114731711669e-07, 2.696057365855835e-07, 2.696057365855835e-07, 5.392114731711669e-07, 2.696057365855835e-07, 6.183623922973797e-06, 1.236724784594759e-05, 6.183623922973797e-06, 6.155093080032972e-06, 1.231018616006594e-05, 6.155093080032972e-06, 5.912112010553704e-06, 1.182422402110741e-05, 5.912112010553704e-06, 5.889056254466219e-06, 1.177811250893244e-05, 5.889056254466219e-06, 6.303551340536771e-06, 1.260710268107354e-05, 6.303551340536771e-06, 6.303551340536771e-06, 1.260710268107354e-05, 6.303551340536771e-06, 5.613354424330193e-03, 1.122670884866039e-02, 5.613354424330193e-03, 7.118094434821825e-03, 1.423618886964365e-02, 7.118094434821825e-03, 7.312603453850276e-03, 1.462520690770055e-02, 7.312603453850276e-03, 1.086632024319961e-02, 2.173264048639922e-02, 1.086632024319961e-02, 5.602703320771599e-03, 1.120540664154320e-02, 5.602703320771599e-03, 5.602703320771602e-03, 1.120540664154320e-02, 5.602703320771602e-03, 7.616565837755218e-02, 1.523313167551044e-01, 7.616565837755218e-02, 6.264552169364081e-02, 1.252910433872816e-01, 6.264552169364081e-02, 7.786047747989822e-02, 1.557209549597964e-01, 7.786047747989822e-02, 5.713721481378458e-05, 1.142744296275692e-04, 5.713721481378458e-05, 1.388271413347298e-01, 2.776542826694595e-01, 1.388271413347298e-01, 1.388271413347298e-01, 2.776542826694595e-01, 1.388271413347298e-01, 2.467439104481722e-03, 4.934878208963443e-03, 2.467439104481722e-03, 3.063244551262781e-03, 6.126489102525562e-03, 3.063244551262781e-03, 2.876152960532595e-02, 5.752305921065190e-02, 2.876152960532595e-02, 7.913220631084264e-02, 1.582644126216853e-01, 7.913220631084264e-02, 1.125838721688566e-02, 2.251677443377132e-02, 1.125838721688566e-02, 1.125838720880113e-02, 2.251677441760227e-02, 1.125838720880113e-02, 1.218576353025488e-02, 2.437152706050976e-02, 1.218576353025488e-02, 1.032166398951879e-02, 2.064332797903758e-02, 1.032166398951879e-02, 1.089811188433333e-02, 2.179622376866666e-02, 1.089811188433333e-02, 1.143430080685923e-02, 2.286860161371846e-02, 1.143430080685923e-02, 1.115884378879000e-02, 2.231768757758000e-02, 1.115884378879000e-02, 1.115884378879000e-02, 2.231768757758000e-02, 1.115884378879000e-02, 1.455428094710289e-02, 2.910856189420578e-02, 1.455428094710289e-02, 7.665003222168145e-03, 1.533000644433629e-02, 7.665003222168145e-03, 8.765359561664026e-03, 1.753071912332805e-02, 8.765359561664026e-03, 1.031030459116937e-02, 2.062060918233874e-02, 1.031030459116937e-02, 9.488108772441259e-03, 1.897621754488252e-02, 9.488108772441259e-03, 9.488108772441259e-03, 1.897621754488252e-02, 9.488108772441259e-03, 5.644102971398867e-03, 1.128820594279773e-02, 5.644102971398867e-03, 4.511731785831075e-02, 9.023463571662149e-02, 4.511731785831075e-02, 3.992095144245824e-02, 7.984190288491648e-02, 3.992095144245824e-02, 3.362138650534369e-02, 6.724277301068737e-02, 3.362138650534369e-02, 3.850323641031246e-02, 7.700647282062492e-02, 3.850323641031246e-02, 3.850323641031249e-02, 7.700647282062498e-02, 3.850323641031249e-02, 1.100773754454750e-02, 2.201547508909499e-02, 1.100773754454750e-02, 2.874627254168770e-02, 5.749254508337540e-02, 2.874627254168770e-02, 4.251154639122009e-02, 8.502309278244018e-02, 4.251154639122009e-02, 5.260854494962382e-02, 1.052170898992476e-01, 5.260854494962382e-02, 1.323543554302621e-01, 2.647087108605241e-01, 1.323543554302621e-01, 1.323543554302622e-01, 2.647087108605244e-01, 1.323543554302622e-01, 8.140154004474657e-03, 1.628030800894931e-02, 8.140154004474657e-03, 3.470455326306633e-03, 6.940910652613265e-03, 3.470455326306633e-03, 4.211162247995764e-03, 8.422324495991527e-03, 4.211162247995764e-03, 1.226253398065706e-01, 2.452506796131412e-01, 1.226253398065706e-01, 1.430707984945069e-02, 2.861415969890137e-02, 1.430707984945069e-02, 1.430707985286488e-02, 2.861415970572976e-02, 1.430707985286488e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05