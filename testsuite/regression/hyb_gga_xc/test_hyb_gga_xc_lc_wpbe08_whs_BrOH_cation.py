
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_lc_wpbe08_whs_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_wpbe08_whs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.074321428887531e+01, -2.074324255133955e+01, -2.074343460159628e+01, -2.074301274959571e+01, -2.074322435003299e+01, -2.074322435003299e+01, -3.262480980929054e+00, -3.262448413550034e+00, -3.261780969520053e+00, -3.263601248225767e+00, -3.262525882890617e+00, -3.262525882890617e+00, -4.887984260733783e-01, -4.884942988155616e-01, -4.830380834077518e-01, -4.879318310148587e-01, -4.886839324215014e-01, -4.886839324215014e-01, -4.957625611151767e-02, -5.096165280512592e-02, -5.963652940713773e-01, -2.746220580793566e-02, -4.394931634106967e-02, -4.394931634106965e-02, -1.706299201707292e-06, -1.995251958654041e-06, -3.849553298468040e-04, -3.254616331694166e-07, -9.515326192881098e-07, -9.515326192881100e-07, -4.840919488185640e+00, -4.840520814552619e+00, -4.840910845306193e+00, -4.840558710281559e+00, -4.840711831755027e+00, -4.840711831755027e+00, -1.883108722672192e+00, -1.893383887306118e+00, -1.883461737120922e+00, -1.892522513364606e+00, -1.888650915201417e+00, -1.888650915201417e+00, -3.951024171317128e-01, -4.396119918516650e-01, -3.554795092547759e-01, -3.732868497520362e-01, -4.037534096536435e-01, -4.037534096536436e-01, -1.032407708021939e-02, -5.421857165502221e-02, -8.099153906717034e-03, -1.641008121973624e+00, -1.601080281540145e-02, -1.601080281540145e-02, -1.498615919015783e-07, -3.040740598596438e-07, -1.374145060452916e-07, -2.067823958752536e-03, -2.844911159299905e-07, -2.844911159363055e-07, -3.978220262019920e-01, -3.932991130379467e-01, -3.947839668185054e-01, -3.960924901880513e-01, -3.954277625012673e-01, -3.954277625012673e-01, -3.841289407130233e-01, -3.133246018871729e-01, -3.304470988232608e-01, -3.495021002074807e-01, -3.394403836335608e-01, -3.394403836335608e-01, -4.676616998736611e-01, -8.644627110331754e-02, -1.174502391016695e-01, -1.814756593012944e-01, -1.454508503616329e-01, -1.454508503616330e-01, -2.767390933308519e-01, -3.294931384290585e-04, -9.585827938909564e-04, -1.703476658474301e-01, -4.767477290780837e-03, -4.767477290780837e-03, -4.780208663160821e-06, -5.800439288993573e-09, -5.412879315840544e-08, -3.841031286509306e-03, -2.190285014491298e-07, -2.190285009895802e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_lc_wpbe08_whs_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_wpbe08_whs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.501615032472843e+01, -2.501612082254033e+01, -2.501626678183040e+01, -2.501620577313922e+01, -2.501660964087556e+01, -2.501670350043463e+01, -2.501529043629922e+01, -2.501502731341176e+01, -2.501621695786994e+01, -2.501567809442495e+01, -2.501621695786994e+01, -2.501567809442495e+01, -3.896429795374443e+00, -3.896618587870894e+00, -3.896478806158567e+00, -3.896675444104303e+00, -3.897743122573151e+00, -3.898058303188892e+00, -3.896206716387227e+00, -3.896528109868044e+00, -3.895688341384563e+00, -3.897681145742134e+00, -3.895688341384563e+00, -3.897681145742134e+00, -6.171301729149027e-01, -6.221718655445093e-01, -6.149084315085995e-01, -6.210799476847018e-01, -5.808797049168057e-01, -5.731152768724520e-01, -5.825159437774899e-01, -5.850425246587687e-01, -6.215613800714885e-01, -5.484218444036159e-01, -6.215613800714885e-01, -5.484218444036159e-01, -7.111353192922115e-02, -7.350588014978239e-02, -7.397614591931845e-02, -7.690115974263534e-02, -7.316621342067633e-01, -7.685169101753394e-01, -3.925990369233705e-02, -3.994347637906810e-02, -5.759134690111066e-02, -3.310506007724613e-02, -5.759134690111060e-02, -3.310506007724610e-02, -3.103690797896263e-06, -3.715880615963827e-06, -3.578185498522744e-06, -4.386406871554687e-06, -7.608588041238520e-04, -9.021900965868504e-04, -6.686559750396980e-07, -6.359305852189551e-07, -2.183883890976857e-06, -4.096472900881254e-07, -2.183883890974584e-06, -4.096472900859546e-07, -6.079340162270257e+00, -6.077820972102412e+00, -6.082352971897977e+00, -6.080738765275702e+00, -6.079503407985117e+00, -6.077921916086818e+00, -6.082093734373233e+00, -6.080567269550191e+00, -6.080883588614112e+00, -6.079287890675821e+00, -6.080883588614112e+00, -6.079287890675821e+00, -2.009647927224311e+00, -2.009537218008845e+00, -2.028193964506481e+00, -2.027577017807377e+00, -1.990653970837154e+00, -1.995157032762273e+00, -2.006408950006696e+00, -2.011169121044167e+00, -2.034218964797102e+00, -2.022072360008784e+00, -2.034218964797102e+00, -2.022072360008784e+00, -5.589376631620359e-01, -5.575505087177299e-01, -6.357667432727904e-01, -6.363015432487203e-01, -4.938801210067771e-01, -5.140954595159923e-01, -5.374453447626559e-01, -5.546556819297982e-01, -5.868851727338970e-01, -5.570314736516158e-01, -5.868851727338970e-01, -5.570314736516159e-01, -1.795103446060301e-02, -1.818635014534674e-02, -6.722267999634229e-02, -6.756816966026474e-02, -1.390848215529054e-02, -1.551421091331276e-02, -2.211541006802215e+00, -2.210626353831884e+00, -2.557374394896330e-02, -2.581625396504682e-02, -2.557374394896330e-02, -2.581625396504682e-02, -2.822230554521432e-07, -3.166260572170533e-07, -5.960584455233889e-07, -6.233886933449959e-07, -2.529472479907733e-07, -2.993717204478946e-07, -4.395075255324713e-03, -4.485997063362366e-03, -2.696365292936384e-07, -6.890830052759392e-07, -2.696365292977753e-07, -6.890830052862094e-07, -5.763010607593643e-01, -5.785764406295405e-01, -5.724879071081292e-01, -5.748211396960942e-01, -5.741281719906405e-01, -5.764580362185717e-01, -5.752684176696414e-01, -5.775480673103693e-01, -5.747271232761800e-01, -5.770309815024274e-01, -5.747271232761800e-01, -5.770309815024274e-01, -5.561248036347595e-01, -5.579572981895723e-01, -4.272270761829797e-01, -4.295316589603878e-01, -4.696649579881779e-01, -4.721726946369919e-01, -5.094318694254232e-01, -5.113225036331338e-01, -4.898968227819921e-01, -4.918443856925475e-01, -4.898968227819921e-01, -4.918443856925475e-01, -6.725813091458710e-01, -6.740793769745107e-01, -1.030669622256250e-01, -1.036625347896175e-01, -1.469092949557055e-01, -1.488497326612180e-01, -2.694984003841181e-01, -2.712078042457076e-01, -2.040036436507980e-01, -2.041650316559411e-01, -2.040036436507979e-01, -2.041650316559411e-01, -3.837329489180779e-01, -3.871392614041115e-01, -7.045266699674201e-04, -7.197879432060741e-04, -1.989059378686107e-03, -2.203514983924711e-03, -2.649659804209311e-01, -2.693183411311935e-01, -9.016752147216674e-03, -9.826948203503834e-03, -9.016752147216568e-03, -9.826948203503881e-03, -9.127514443701013e-06, -1.013993549590861e-05, -1.157867597742420e-08, -1.165804442421362e-08, -9.776060972150655e-08, -1.175173813869792e-07, -7.682348279861301e-03, -7.992014065544921e-03, -2.293385697986880e-07, -5.311813945786717e-07, -2.293385682672357e-07, -5.311813945807743e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_lc_wpbe08_whs_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_wpbe08_whs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.391087483296361e-08, 3.855567579455844e-10, -1.391095299326063e-08, -1.391075824390792e-08, 3.855637563905792e-10, -1.391086729695848e-08, -1.391026317116461e-08, 3.855734091993355e-10, -1.391019594781180e-08, -1.391159543053006e-08, 3.854697494807899e-10, -1.391187876779106e-08, -1.391082869531204e-08, 3.855271085313426e-10, -1.391110361772171e-08, -1.391082869531204e-08, 3.855271085313426e-10, -1.391110361772171e-08, -1.782454194500572e-05, 2.158879157656427e-06, -1.783005283030556e-05, -1.782464577834902e-05, 2.159323527660124e-06, -1.783058764281431e-05, -1.783187238080244e-05, 2.169371669283230e-06, -1.783464276275927e-05, -1.780819173664349e-05, 2.149943255801801e-06, -1.781206813405054e-05, -1.783523220069303e-05, 2.159156568071321e-06, -1.781675571258411e-05, -1.783523220069303e-05, 2.159156568071321e-06, -1.781675571258411e-05, -8.433475542870347e-03, 3.586683876242116e-03, -8.358822413637536e-03, -8.479427388742701e-03, 3.559577090514688e-03, -8.388411203258164e-03, -9.225957627135604e-03, 2.876045381924161e-03, -9.312161151692824e-03, -9.024807218765835e-03, 2.810968523930000e-03, -9.001000464454130e-03, -8.546296385956097e-03, 2.852692094259074e-03, -9.098757922385455e-03, -8.546296385956097e-03, 2.852692094259074e-03, -9.098757922385455e-03, -4.085641873782799e-01, 3.108902575321890e-01, -3.858385864754644e-01, -4.000523615540188e-01, 3.280826843588528e-01, -3.709053911041488e-01, -5.199731225348083e-03, 1.940414170439823e-03, -4.656960555477703e-03, -4.893921626259521e-01, 2.258949661124237e-01, -4.831572685501293e-01, -4.222074418733759e-01, 1.882246441728870e-01, -4.046881197318687e-01, -4.222074418733768e-01, 1.882246441728873e-01, -4.046881197318713e-01, 4.233492221023594e-03, 8.471332723898200e-03, 4.232745769814701e-03, 4.979361126433320e-03, 9.964515393423459e-03, 4.978172343717360e-03, 1.064773816805741e-02, 5.126941006152715e-02, 6.038024851751589e-03, 1.504165672557061e-03, 3.008683449168684e-03, 1.504183032981078e-03, 2.206049039127003e-03, 4.414566270491809e-03, 2.207055067589888e-03, 2.206049039575652e-03, 4.414566270088269e-03, 2.207055068039021e-03, -3.846255551001725e-06, 5.666948009101149e-07, -3.850068398124786e-06, -3.844551953604705e-06, 5.704413223836728e-07, -3.848360800137743e-06, -3.846159726713422e-06, 5.668443577206054e-07, -3.849977451235371e-06, -3.844652341125502e-06, 5.701514261968389e-07, -3.848474853544706e-06, -3.845415031874265e-06, 5.686004611431787e-07, -3.849204264124432e-06, -3.845415031874265e-06, 5.686004611431787e-07, -3.849204264124432e-06, -1.398198831333345e-04, 1.221811741084532e-05, -1.398455307705220e-04, -1.369601941034410e-04, 1.218856108878283e-05, -1.370839054305377e-04, -1.399935496642275e-04, 1.161795009824474e-05, -1.401057026495210e-04, -1.375812678087334e-04, 1.159619612758267e-05, -1.376136612692828e-04, -1.379720632387002e-04, 1.250231835492394e-05, -1.382498035256210e-04, -1.379720632387002e-04, 1.250231835492394e-05, -1.382498035256210e-04, -1.110957197727000e-02, 1.180452397706796e-02, -1.126598927575296e-02, -5.419727406352982e-03, 1.489938286408802e-02, -5.394913015271334e-03, -1.589390304426897e-02, 1.535577972946644e-02, -1.324331195812567e-02, -9.223225441083022e-03, 2.278859644535717e-02, -7.095573662840308e-03, -9.013850113246340e-03, 1.179162458306085e-02, -1.185493378609448e-02, -9.013850113246355e-03, 1.179162458306085e-02, -1.185493378609449e-02, -3.164227284468099e-01, 1.261488607621168e-01, -3.291990805604617e-01, -3.508468979579290e-01, 1.139695105238702e-01, -3.487360153139142e-01, -2.704801152148154e-01, 1.279112469154852e-01, -3.045467580411440e-01, -1.577318317246658e-04, 1.207887057881010e-04, -1.581135880973124e-04, -4.027585863393993e-01, 2.409472748596690e-01, -5.310175702955491e-01, -4.027585863393993e-01, 2.409472748596690e-01, -5.310175702955491e-01, 1.466704533410495e-03, 2.933546798259037e-03, 1.466702888322456e-03, 1.849393379303711e-03, 3.699144102256803e-03, 1.849395481840089e-03, 1.842526057220369e-02, 3.685108437354808e-02, 1.842514028042408e-02, -8.985526641656141e-02, 1.250822448857004e-01, -8.819917311632108e-02, 7.051649728216688e-03, 1.410360645889085e-02, 7.051186393992825e-03, 7.051649731495812e-03, 1.410360646869916e-02, 7.051186397272904e-03, -4.066761428789804e-03, 2.514346072918196e-02, -3.846590678014010e-03, -6.412616597979762e-03, 2.157065992117518e-02, -6.183592072813857e-03, -5.666579446262884e-03, 2.270457728491302e-02, -5.439274311684365e-03, -4.988009812075308e-03, 2.373865089872573e-02, -4.765617669103880e-03, -5.334650957408840e-03, 2.321019190891285e-02, -5.109783127899187e-03, -5.334650957408840e-03, 2.321019190891285e-02, -5.109783127899187e-03, -3.205364471017595e-03, 2.977269840520045e-02, -3.005778355032272e-03, -2.243215868598269e-02, 1.592277215307748e-02, -2.201498091600625e-02, -1.729844472480252e-02, 1.838749842245184e-02, -1.692302066135627e-02, -1.218111157871117e-02, 2.169730037966074e-02, -1.191368006151023e-02, -1.477410401073565e-02, 1.995497716545398e-02, -1.447315984513376e-02, -1.477410401073565e-02, 1.995497716545398e-02, -1.447315984513376e-02, -5.052245703201739e-03, 1.183264719377924e-02, -4.985409195493663e-03, -2.426348087238243e-01, 8.498631964250428e-02, -2.403308353169050e-01, -1.678788541696407e-01, 7.821972418059023e-02, -1.638818635262684e-01, -5.688785257875478e-02, 6.964082434152352e-02, -5.536407484461885e-02, -1.018752542521346e-01, 7.796350422599263e-02, -1.019806753599968e-01, -1.018752542521352e-01, 7.796350422599266e-02, -1.019806753599972e-01, -2.867317142826722e-02, 2.285950305464996e-02, -2.781562008213723e-02, 8.521658549188099e-03, 4.213028450134424e-02, 8.080488313528929e-03, -1.640603450777756e-02, 6.458066222247318e-02, -2.485214479855032e-02, -4.702965825163791e-02, 1.099497136873529e-01, -4.268269657736245e-02, -1.965879669918805e-01, 2.182010199987041e-01, -2.944309973627358e-01, -1.965879669918766e-01, 2.182010199987039e-01, -2.944309973627397e-01, 5.270390721653615e-03, 1.056259116938041e-02, 5.267848006324198e-03, 2.005763201644689e-03, 4.011531506685853e-03, 2.005761204792090e-03, 2.510353023187405e-03, 5.020763908060611e-03, 2.510342048562026e-03, -1.954819936634262e-01, 2.002688123716920e-01, -2.045770731556706e-01, 9.000693466761781e-03, 1.800188742308661e-02, 9.000511787764807e-03, 9.000693470464172e-03, 1.800188743452388e-02, 9.000511791429758e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05