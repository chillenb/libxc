
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_pbe_sol0_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbe_sol0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.541741592251613e+01, -1.541744569052040e+01, -1.541761219835980e+01, -1.541716860991981e+01, -1.541739618298212e+01, -1.541739618298212e+01, -2.579659890070888e+00, -2.579649619383851e+00, -2.579474528746748e+00, -2.580260017483948e+00, -2.579708769863706e+00, -2.579708769863706e+00, -5.350723637118773e-01, -5.346727513256098e-01, -5.259771726630195e-01, -5.298417649373796e-01, -5.303993352923950e-01, -5.303993352923950e-01, -1.585412743689185e-01, -1.601671184487076e-01, -6.173861062138807e-01, -1.298177337022589e-01, -1.496470862453553e-01, -1.496470862453552e-01, -7.575824677424252e-03, -7.976767925592512e-03, -4.282678524786235e-02, -4.371146983394387e-03, -6.098872234093718e-03, -6.098872234093718e-03, -3.779806972553051e+00, -3.780010788610028e+00, -3.779821161235077e+00, -3.780000990088131e+00, -3.779907456346423e+00, -3.779907456346423e+00, -1.528807310101082e+00, -1.537408568149167e+00, -1.526140643613559e+00, -1.533707120154606e+00, -1.534906822958423e+00, -1.534906822958423e+00, -4.686963019761108e-01, -5.071441229499327e-01, -4.365870205500851e-01, -4.532514701648745e-01, -4.760073074499790e-01, -4.760073074499790e-01, -9.964637072529528e-02, -1.650718904058638e-01, -9.345096277916314e-02, -1.426973167446542e+00, -1.105133372703818e-01, -1.105133372703818e-01, -3.375018281050846e-03, -4.273008729884080e-03, -3.271769466445046e-03, -6.627334947241841e-02, -4.109635873512730e-03, -4.109635873512735e-03, -4.733965504527408e-01, -4.697165840923667e-01, -4.709617776306361e-01, -4.720283666205458e-01, -4.714900538659141e-01, -4.714900538659141e-01, -4.620793494016258e-01, -4.002851253607571e-01, -4.162183564320773e-01, -4.332683551359494e-01, -4.243844143389696e-01, -4.243844143389696e-01, -5.295199323939113e-01, -1.995055036134514e-01, -2.305369109529263e-01, -2.906622531388879e-01, -2.577690400004967e-01, -2.577690400004967e-01, -3.708371527751939e-01, -4.107425327763710e-02, -5.492385175801627e-02, -2.806346138277463e-01, -8.064743945931935e-02, -8.064743945931935e-02, -1.066986878586662e-02, -1.142421177650491e-03, -2.402236224880085e-03, -7.661368738502548e-02, -3.774197264140168e-03, -3.774197264140162e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_pbe_sol0_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbe_sol0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.936293258597375e+01, -1.936290943658273e+01, -1.936301354990758e+01, -1.936296859782671e+01, -1.936326094152718e+01, -1.936332451487229e+01, -1.936234228414870e+01, -1.936215879117175e+01, -1.936297743306766e+01, -1.936261860566914e+01, -1.936297743306766e+01, -1.936261860566914e+01, -3.217726529352318e+00, -3.217823850798006e+00, -3.217758085507201e+00, -3.217858424933238e+00, -3.218551730114610e+00, -3.218746127375313e+00, -3.217681192007977e+00, -3.217874689581635e+00, -3.217183837121619e+00, -3.218595372332656e+00, -3.217183837121619e+00, -3.218595372332656e+00, -6.686201025632033e-01, -6.719931859668147e-01, -6.670928197810641e-01, -6.712241226542006e-01, -6.429983547539599e-01, -6.375452164420093e-01, -6.448218077496679e-01, -6.465853540973300e-01, -6.710660985212300e-01, -6.183779398738097e-01, -6.710660985212300e-01, -6.183779398738097e-01, -1.803375822533952e-01, -1.840292794123564e-01, -1.837689797444679e-01, -1.881520168569862e-01, -7.621084194636734e-01, -7.880154215187192e-01, -1.359855375365248e-01, -1.370153307328222e-01, -1.641719932810134e-01, -1.278887010296650e-01, -1.641719932810133e-01, -1.278887010296649e-01, -9.748245165325875e-03, -1.035472445977549e-02, -1.021102691784369e-02, -1.093309533723248e-02, -5.281964563019519e-02, -5.527411128750057e-02, -5.871705390814688e-03, -5.774150026890656e-03, -8.701790878246381e-03, -4.957745964173865e-03, -8.701790878246383e-03, -4.957745964173865e-03, -4.859307061105723e+00, -4.858179701781335e+00, -4.861140824137431e+00, -4.859957059425862e+00, -4.859407783077915e+00, -4.858242899158789e+00, -4.860985419760574e+00, -4.859853254376350e+00, -4.860245092470326e+00, -4.859073432801356e+00, -4.860245092470326e+00, -4.859073432801356e+00, -1.766671176770588e+00, -1.766585365588390e+00, -1.781793668466762e+00, -1.781310145877129e+00, -1.749542419016186e+00, -1.753704819481198e+00, -1.762615619687414e+00, -1.766908772745021e+00, -1.787354673092569e+00, -1.777035750766538e+00, -1.787354673092569e+00, -1.777035750766538e+00, -6.108569984793828e-01, -6.097972968346098e-01, -6.665281980040285e-01, -6.669135478196112e-01, -5.592024989500084e-01, -5.744159192127076e-01, -5.892359096910771e-01, -6.020002059666587e-01, -6.317675382496608e-01, -6.093596197467356e-01, -6.317675382496609e-01, -6.093596197467357e-01, -1.070760715397792e-01, -1.071561840113622e-01, -1.749462555494212e-01, -1.754082607208017e-01, -1.004103255910765e-01, -1.028149606114786e-01, -1.880776073068892e+00, -1.880085039960326e+00, -1.164991586393738e-01, -1.149716718407565e-01, -1.164991586393738e-01, -1.149716718407565e-01, -4.406452954447141e-03, -4.579356504235519e-03, -5.649508214359915e-03, -5.735074635003359e-03, -4.223025900957131e-03, -4.471150139510998e-03, -7.721613560069995e-02, -7.777560122618840e-02, -4.314828575830726e-03, -5.920161267310992e-03, -4.314828575830731e-03, -5.920161267310995e-03, -6.191201048778323e-01, -6.207940593367784e-01, -6.163176401961461e-01, -6.180407832189710e-01, -6.174487934142897e-01, -6.191678085413351e-01, -6.182803312448945e-01, -6.199596785902732e-01, -6.178786279567373e-01, -6.195772204870612e-01, -6.178786279567373e-01, -6.195772204870612e-01, -6.033133988632020e-01, -6.046580213428540e-01, -5.082873762204918e-01, -5.100172801089458e-01, -5.390391770790485e-01, -5.409229796827760e-01, -5.678779517263915e-01, -5.692959781032300e-01, -5.536696073784027e-01, -5.551404622788600e-01, -5.536696073784027e-01, -5.551404622788600e-01, -6.958368038866032e-01, -6.969128575109987e-01, -2.202602417932396e-01, -2.210110918007865e-01, -2.693672912971863e-01, -2.711969543316127e-01, -3.709953299408561e-01, -3.723761864019647e-01, -3.182060825957540e-01, -3.182338561298663e-01, -3.182060825957540e-01, -3.182338561298663e-01, -4.712121161328566e-01, -4.737777796049409e-01, -5.211634736968612e-02, -5.241951972585833e-02, -6.659482163667413e-02, -6.815812066931345e-02, -3.635599661623860e-01, -3.671942669769557e-01, -8.868909329815623e-02, -8.870240331323162e-02, -8.868909329815620e-02, -8.870240331323159e-02, -1.390934713220358e-02, -1.440676187675828e-02, -1.521345873212833e-03, -1.524746789240336e-03, -3.094553446832347e-03, -3.291041944032081e-03, -8.507425535641980e-02, -8.591497457747910e-02, -4.085805455864451e-03, -5.427817332680687e-03, -4.085805455864445e-03, -5.427817332680680e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_pbe_sol0_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbe_sol0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-6.137037838152969e-09, 3.924232368725627e-10, -6.137071881747679e-09, -6.136970596806608e-09, 3.924296816115061e-10, -6.137022140922030e-09, -6.136721844419701e-09, 3.924376028377716e-10, -6.136678607355586e-09, -6.137493576940366e-09, 3.923421653813450e-10, -6.137648830189807e-09, -6.137015737308766e-09, 3.923951125479222e-10, -6.137224169506012e-09, -6.137015737308766e-09, 3.923951125479222e-10, -6.137224169506012e-09, -7.657014937951732e-06, 2.160283788062830e-06, -7.658757628145726e-06, -7.656866406191704e-06, 2.160679371320279e-06, -7.658757270209678e-06, -7.655095994286600e-06, 2.169596379790034e-06, -7.655596347948643e-06, -7.653520896951999e-06, 2.152143033245858e-06, -7.654423451218859e-06, -7.662251469398209e-06, 2.160504755167462e-06, -7.651915015603807e-06, -7.662251469398209e-06, 2.160504755167462e-06, -7.651915015603807e-06, -4.126600343633189e-03, 3.502464509038655e-03, -4.091288101657728e-03, -4.152355890165178e-03, 3.484212993337901e-03, -4.109147917646373e-03, -4.615032443243398e-03, 2.998335795936386e-03, -4.667260941864389e-03, -4.506240154638763e-03, 2.923016200794973e-03, -4.492051236376960e-03, -4.266914838873843e-03, 2.961848518020074e-03, -4.651914209364691e-03, -4.266914838873843e-03, 2.961848518020074e-03, -4.651914209364691e-03, -6.301163780277783e-01, 4.036523593851310e-01, -5.584071405073090e-01, -6.136816195640068e-01, 4.167732807198712e-01, -5.303259406083251e-01, -2.454128563206749e-03, 1.877927402871148e-03, -2.142889497890937e-03, -1.170967187679827e+00, 3.631115076261158e-01, -1.121498799509223e+00, -5.671440043656060e-01, 2.895021812397551e-01, -2.145283085956192e+00, -5.671440043656056e-01, 2.895021812397556e-01, -2.145283085956194e+00, -6.513912037031393e+00, 1.780923863322236e-02, -6.446925010459978e+00, -6.860516576968964e+00, 2.094603145755870e-02, -6.823125031361358e+00, -3.570552338493343e+00, 1.058875238306064e-01, -3.568063076657904e+00, -5.979328780431069e+00, 6.328648217783347e-03, -5.817115135848520e+00, -6.509951779000637e+00, 9.284352922054217e-03, -1.653760444237766e+01, -6.509951779000424e+00, 9.284352921806719e-03, -1.653760444237747e+01, -1.561297822554197e-06, 5.447862516555482e-07, -1.563019276122651e-06, -1.559188770522987e-06, 5.480625369107096e-07, -1.560926135928459e-06, -1.561195168073303e-06, 5.449167399912604e-07, -1.562930335252762e-06, -1.559345253664166e-06, 5.478087633638220e-07, -1.561071108887445e-06, -1.560232154525921e-06, 5.464531415813659e-07, -1.561958999948857e-06, -1.560232154525921e-06, 5.464531415813659e-07, -1.561958999948857e-06, -6.769686408100434e-05, 1.379628549525131e-05, -6.771074874827853e-05, -6.601508005578435e-05, 1.369758735885949e-05, -6.608436197490599e-05, -6.862012372801285e-05, 1.326989552314756e-05, -6.845923341636442e-05, -6.716098200685986e-05, 1.318810272778560e-05, -6.697118286157702e-05, -6.609003778246048e-05, 1.400297483065355e-05, -6.665630247921529e-05, -6.609003778246048e-05, 1.400297483065355e-05, -6.665630247921529e-05, -5.810503453061082e-03, 1.016619295788007e-02, -5.929671449050485e-03, -2.826692690596289e-03, 1.125460881018527e-02, -2.818087663760582e-03, -9.097386775059616e-03, 1.338680806093630e-02, -7.111002483071668e-03, -5.816449732214499e-03, 1.749442325527369e-02, -4.018195470914414e-03, -4.420135966070436e-03, 1.001437922971996e-02, -6.513223985261513e-03, -4.420135966070438e-03, 1.001437922971996e-02, -6.513223985261510e-03, -1.828926544869014e+00, 2.363423791386441e-01, -1.845585965125423e+00, -4.825460384524075e-01, 1.722295977524940e-01, -4.756216409045883e-01, -2.157603772874922e+00, 2.428965328721264e-01, -2.039553791792422e+00, -5.306022493919093e-05, 9.942030332770422e-05, -5.324936014001047e-05, -1.712545886389271e+00, 4.172491666969919e-01, -1.903627484218133e+00, -1.712545886389271e+00, 4.172491666969919e-01, -1.903627484218133e+00, -8.397166547451340e+00, 6.171121297633935e-03, -7.267610811766250e+00, -7.227648751088783e+00, 7.780920747421459e-03, -6.672868099959221e+00, -4.114311536622139e+01, 7.750279055850653e-02, -4.576549255484042e+01, -3.634829280188069e+00, 2.494517602975471e-01, -3.501202297809646e+00, -2.046925053739801e+01, 2.966254161607970e-02, -2.009364877818641e+01, -2.046925053739791e+01, 2.966254161369759e-02, -2.009364877818636e+01, -2.877450245718813e-03, 1.792659001284867e-02, -2.699345433408044e-03, -3.842185704892098e-03, 1.614886381961827e-02, -3.663854751625463e-03, -3.525313139895120e-03, 1.672929276764286e-02, -3.346915120863903e-03, -3.245335608896989e-03, 1.724549016285097e-02, -3.067287006269383e-03, -3.387433493591847e-03, 1.698327859994344e-02, -3.209198689688190e-03, -3.387433493591847e-03, 1.698327859994344e-02, -3.209198689688190e-03, -2.838503081093494e-03, 2.085943275450910e-02, -2.665014379799978e-03, -1.292655353982374e-02, 1.502913877693196e-02, -1.262388017420788e-02, -1.002069578160933e-02, 1.614687317543641e-02, -9.741759883863631e-03, -7.190326701142861e-03, 1.764044252227230e-02, -6.976536737235665e-03, -8.612846612177774e-03, 1.687721053499117e-02, -8.374280816203186e-03, -8.612846612177774e-03, 1.687721053499117e-02, -8.374280816203186e-03, -2.478965128791219e-03, 9.031795125911575e-03, -2.442133244627560e-03, -2.397191200679457e-01, 1.154815614433179e-01, -2.354506084536857e-01, -1.353933479250671e-01, 9.336554331022104e-02, -1.310248452307666e-01, -4.720171904822432e-02, 6.569618122322450e-02, -4.563589541096368e-02, -8.119783122448526e-02, 8.207722746589632e-02, -8.169346767533178e-02, -8.119783122448533e-02, 8.207722746589632e-02, -8.169346767533185e-02, -1.766742222170336e-02, 2.152487737361528e-02, -1.702516225910292e-02, -3.284859413790874e+00, 8.721472124114714e-02, -3.286437439617701e+00, -3.087275202839738e+00, 1.318486983034387e-01, -3.132968792658479e+00, -5.178103554404739e-02, 9.650295197887153e-02, -4.656605856833621e-02, -3.240030841952604e+00, 4.165989187035299e-01, -3.641180382319080e+00, -3.240030841952605e+00, 4.165989187035304e-01, -3.641180382319082e+00, -5.149530564256245e+00, 2.219217594856401e-02, -5.255540263198943e+00, -2.585131228334872e+01, 8.440029620734958e-03, -4.579341845839338e+01, -1.594883005815394e+01, 1.056237513262514e-02, -1.697803989861236e+01, -3.681242789675127e+00, 3.870874530572532e-01, -3.581368364522398e+00, -4.220468557501050e+01, 3.786174589046336e-02, -2.086015700883662e+01, -4.220468557500728e+01, 3.786174589577986e-02, -2.086015700883338e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05