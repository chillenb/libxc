
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_m06_2x_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-8.314621287659840e+00, -8.314564249815406e+00, -8.314364833903225e+00, -8.315207691508398e+00, -8.314758560282879e+00, -8.314758560282879e+00, -1.682220486015499e+00, -1.682206368869699e+00, -1.681902073015223e+00, -1.682865540260658e+00, -1.682283561491128e+00, -1.682283561491128e+00, -3.524019123030586e-01, -3.532270950401029e-01, -3.755595291206361e-01, -3.679236953943668e-01, -3.704886925834384e-01, -3.704886925834384e-01, -1.274166351853688e-01, -1.261325179717256e-01, -4.859030664667728e-01, -1.233898090854226e-01, -1.341927463205614e-01, -1.341927463205615e-01, -1.685978974045552e-02, -1.774307421577920e-02, -8.739086059347898e-02, -9.754768680223331e-03, -1.358957511552544e-02, -1.358957511552543e-02, -1.797745452060719e+00, -1.794562197441145e+00, -1.797566078444521e+00, -1.794757547886709e+00, -1.796135502757704e+00, -1.796135502757704e+00, -1.009145950055049e+00, -1.018039004297794e+00, -1.006581139429011e+00, -1.014449209230240e+00, -1.015370851846026e+00, -1.015370851846026e+00, -2.595931081623525e-01, -2.352028389877013e-01, -2.595773299629192e-01, -2.394858511179944e-01, -2.563775418012738e-01, -2.563775418012738e-01, -1.338114221926194e-01, -1.459655435837569e-01, -1.324942458846474e-01, -7.816603571966561e-01, -1.154290689119806e-01, -1.154290689119800e-01, -7.536071587940823e-03, -9.536995096751360e-03, -7.297434566969546e-03, -1.158767381220527e-01, -9.163569138513277e-03, -9.163569138513317e-03, -1.843021222722795e-01, -2.036522823732206e-01, -2.048102917127688e-01, -1.997161200128934e-01, -2.031417914658772e-01, -2.031417914658774e-01, -1.851753988031874e-01, -2.400376835472782e-01, -2.227905442633136e-01, -2.022461587331362e-01, -2.151285614399497e-01, -2.151285614399497e-01, -2.593517457750248e-01, -1.660178247287251e-01, -1.655835744129004e-01, -1.742765589241134e-01, -1.597060493750488e-01, -1.597060493750488e-01, -2.280834161337461e-01, -8.439586772285401e-02, -1.061983108026398e-01, -1.648654654177962e-01, -1.157892887775998e-01, -1.157892887775990e-01, -2.371636281741667e-02, -2.552037852917223e-03, -5.363554035714677e-03, -1.136401648785457e-01, -8.417437696823663e-03, -8.417437696823588e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_m06_2x_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.329568060563060e+00, -6.327712608538315e+00, -6.329337008393631e+00, -6.327545183944437e+00, -6.328677094986188e+00, -6.326578027516036e+00, -6.331266776641501e+00, -6.329883162934640e+00, -6.329419244063409e+00, -6.328621315260952e+00, -6.329419244063409e+00, -6.328621315260952e+00, -1.982412731368784e+00, -1.983071302400671e+00, -1.981731198302665e+00, -1.982638721652870e+00, -1.968467577712470e+00, -1.966120630062324e+00, -1.972911114717572e+00, -1.973046076673278e+00, -1.983597714555833e+00, -1.960966103241154e+00, -1.983597714555833e+00, -1.960966103241154e+00, -1.662004771147171e-01, -1.766344540347160e-01, -1.633988520645706e-01, -1.750181696756954e-01, -1.447070579569306e-01, -1.435352601898529e-01, -1.418017023179496e-01, -1.441802239765119e-01, -1.870457872048526e-01, -1.496809061220758e-01, -1.870457872048526e-01, -1.496809061220758e-01, -7.995342196353206e-02, -5.474092879376133e-02, -7.341650335648775e-02, -4.636882621224680e-02, -4.310037098475973e-01, -3.389814186274929e-01, -6.018535213249564e-02, -7.388999333386274e-02, -8.172379214616925e-02, -5.149453040757266e-02, -8.172379214616755e-02, -5.149453040755387e-02, -2.161041006462754e-02, -2.293135521153275e-02, -2.261588961730305e-02, -2.418243227743867e-02, -9.744763077456502e-02, -9.907631328846329e-02, -1.308699058267467e-02, -1.287029155460457e-02, -1.932465967832803e-02, -1.104891772770344e-02, -1.932465967832803e-02, -1.104891772770666e-02, -1.590599767381711e+00, -1.590304126217166e+00, -1.610803864734989e+00, -1.609819821753857e+00, -1.591967256932541e+00, -1.591154172391206e+00, -1.609290615948239e+00, -1.608870591430463e+00, -1.600653139629432e+00, -1.599985705244264e+00, -1.600653139629432e+00, -1.599985705244264e+00, -1.212169972102532e+00, -1.212968116249544e+00, -1.222665239082696e+00, -1.221819277843552e+00, -1.189475649496184e+00, -1.199026051381058e+00, -1.211943826651516e+00, -1.216670363743745e+00, -1.222750641964096e+00, -1.220172590433900e+00, -1.222750641964096e+00, -1.220172590433900e+00, -2.042553784385014e-01, -2.044604221877612e-01, -1.932968648668748e-01, -1.862159563446860e-01, -2.898344208580070e-01, -2.263754719267746e-01, -2.068150742800688e-01, -2.179985256199450e-01, -2.281825244957164e-01, -2.051116510592255e-01, -2.281825244957164e-01, -2.051116510592255e-01, -5.465884241607918e-02, -5.335890052995601e-02, -1.224476434309266e-01, -1.228950594260323e-01, -6.638396157810325e-02, -5.741296294223067e-02, -7.594249882567272e-01, -7.592659731173048e-01, -4.044945275106247e-02, -5.137097965027708e-02, -4.044945275106741e-02, -5.137097965029413e-02, -9.833973734815888e-03, -1.021936657055556e-02, -1.259531922244678e-02, -1.278707446244032e-02, -9.404637888673548e-03, -9.951760108544223e-03, -9.472627158146818e-02, -9.064005248630663e-02, -9.627390691616699e-03, -1.316123224476917e-02, -9.627390691615244e-03, -1.316123224476822e-02, -1.582043386333499e-01, -1.599413077513530e-01, -3.002511943467863e-01, -2.979796115723364e-01, -2.552614504141855e-01, -2.488476604901823e-01, -1.980375299931427e-01, -1.927665734696584e-01, -2.272851935818192e-01, -2.209083719490773e-01, -2.272851935818192e-01, -2.209083719490785e-01, -1.616287488269461e-01, -1.590841245167711e-01, -1.805501069456350e-01, -1.783218153430319e-01, -1.858684169019126e-01, -1.880898769306926e-01, -1.474052893942224e-01, -1.469610108219233e-01, -1.882491483818972e-01, -1.874411197536695e-01, -1.882491483818972e-01, -1.874411197536695e-01, -2.495231062986939e-01, -2.364554708379031e-01, -1.002998236904947e-01, -9.817740795345735e-02, -4.899190658134099e-02, -4.858454358897678e-02, -2.144296826477192e-01, -2.158671769133303e-01, -1.229159253175723e-01, -1.250331937768807e-01, -1.229159253175724e-01, -1.250331937768806e-01, -2.442426741338090e-01, -2.310971889977424e-01, -9.625459611445340e-02, -9.677456311149582e-02, -1.073047989866198e-01, -1.064661646093195e-01, -1.903453627013667e-01, -1.489201413019593e-01, -6.525242228006016e-02, -4.961779264297441e-02, -6.525242228006373e-02, -4.961779264299485e-02, -3.075057639789851e-02, -3.182972172528666e-02, -3.398277920912191e-03, -3.405947566303547e-03, -6.905446685097334e-03, -7.342539738286016e-03, -6.401729385800670e-02, -5.856945710079543e-02, -9.110717652909960e-03, -1.207524908209912e-02, -9.110717652909918e-03, -1.207524908209934e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m06_2x_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-5.598266800196688e-09, 0.000000000000000e+00, -5.597936581554478e-09, -5.598168598279446e-09, 0.000000000000000e+00, -5.597865282270119e-09, -5.597823716623337e-09, 0.000000000000000e+00, -5.597379641511553e-09, -5.598929934536458e-09, 0.000000000000000e+00, -5.598790626377144e-09, -5.598208412049442e-09, 0.000000000000000e+00, -5.598204358500934e-09, -5.598208412049442e-09, 0.000000000000000e+00, -5.598204358500934e-09, -9.214798807531645e-06, 0.000000000000000e+00, -9.217553442085285e-06, -9.214996720315910e-06, 0.000000000000000e+00, -9.217958119606233e-06, -9.221693624268909e-06, 0.000000000000000e+00, -9.223170724224394e-06, -9.204718485293445e-06, 0.000000000000000e+00, -9.206704512094977e-06, -9.220001687811733e-06, 0.000000000000000e+00, -9.211727341612490e-06, -9.220001687811733e-06, 0.000000000000000e+00, -9.211727341612490e-06, -6.159364141764607e-03, 0.000000000000000e+00, -6.097977054823492e-03, -6.182206253116463e-03, 0.000000000000000e+00, -6.111760888319734e-03, -6.520948353134852e-03, 0.000000000000000e+00, -6.557559390341013e-03, -6.211330186581631e-03, 0.000000000000000e+00, -6.168970015300938e-03, -6.017446813728867e-03, 0.000000000000000e+00, -6.340046951196008e-03, -6.017446813728867e-03, 0.000000000000000e+00, -6.340046951196008e-03, -7.779636114530777e-01, 0.000000000000000e+00, -7.008884930436451e-01, -7.744774465471037e-01, 0.000000000000000e+00, -6.801747854694364e-01, -4.320323409792083e-03, 0.000000000000000e+00, -3.873933165330140e-03, -1.176399687424905e+00, 0.000000000000000e+00, -1.098320792486946e+00, -6.695792692636702e-01, 0.000000000000000e+00, -2.350446607883874e+00, -6.695792692636700e-01, 0.000000000000000e+00, -2.350446607883887e+00, -8.179076424857215e+00, 0.000000000000000e+00, -8.092820319779179e+00, -8.613773530675575e+00, 0.000000000000000e+00, -8.563748417780392e+00, -4.360562556180268e+00, 0.000000000000000e+00, -4.325899935672134e+00, -7.512494978953645e+00, 0.000000000000000e+00, -7.308815195960960e+00, -8.172759622953723e+00, 0.000000000000000e+00, -2.077393877797962e+01, -8.172759622953746e+00, 0.000000000000000e+00, -2.077393877797952e+01, -1.482006990004468e-06, 0.000000000000000e+00, -1.483360706968866e-06, -1.479677952516036e-06, 0.000000000000000e+00, -1.481107252981451e-06, -1.481828408173929e-06, 0.000000000000000e+00, -1.483248030363676e-06, -1.479831604815336e-06, 0.000000000000000e+00, -1.481202510063903e-06, -1.480839916151736e-06, 0.000000000000000e+00, -1.482225909504088e-06, -1.480839916151736e-06, 0.000000000000000e+00, -1.482225909504088e-06, -6.997489017346525e-05, 0.000000000000000e+00, -7.001724473950567e-05, -6.889012631979052e-05, 0.000000000000000e+00, -6.896972753095235e-05, -6.964315574419017e-05, 0.000000000000000e+00, -6.979820722771409e-05, -6.873194171591594e-05, 0.000000000000000e+00, -6.885670854610541e-05, -6.954644860488093e-05, 0.000000000000000e+00, -6.950491056500139e-05, -6.954644860488093e-05, 0.000000000000000e+00, -6.950491056500139e-05, -1.074707096334139e-02, 0.000000000000000e+00, -1.081580196247184e-02, -7.700962889240808e-03, 0.000000000000000e+00, -7.607494640216709e-03, -1.661303568960173e-02, 0.000000000000000e+00, -1.432946972978698e-02, -1.522316858339553e-02, 0.000000000000000e+00, -1.289253479559117e-02, -9.017856606126183e-03, 0.000000000000000e+00, -1.133297994787410e-02, -9.017856606126185e-03, 0.000000000000000e+00, -1.133297994787410e-02, -1.874963481946536e+00, 0.000000000000000e+00, -1.890285319267211e+00, -4.923443653343464e-01, 0.000000000000000e+00, -4.874853504747375e-01, -2.304867340264918e+00, 0.000000000000000e+00, -2.110926841986746e+00, -1.029631045251630e-04, 0.000000000000000e+00, -1.031404739239098e-04, -1.671002230613781e+00, 0.000000000000000e+00, -1.752116909048181e+00, -1.671002230613781e+00, 0.000000000000000e+00, -1.752116909048163e+00, -1.055295090251466e+01, 0.000000000000000e+00, -9.133752216365780e+00, -9.082202493960782e+00, 0.000000000000000e+00, -8.385691631120176e+00, -5.170423139655850e+01, 0.000000000000000e+00, -5.750070932038202e+01, -4.212085449614848e+00, 0.000000000000000e+00, -3.983003864870525e+00, -2.573517791510944e+01, 0.000000000000000e+00, -2.523321213909086e+01, -2.573517791510960e+01, 0.000000000000000e+00, -2.523321213909090e+01, -9.361985875630490e-03, 0.000000000000000e+00, -9.150781543468534e-03, -1.016809027475960e-02, 0.000000000000000e+00, -1.004326326943899e-02, -1.028617807340614e-02, 0.000000000000000e+00, -1.012561621583883e-02, -1.008442313074203e-02, 0.000000000000000e+00, -9.890491189927179e-03, -1.023026634610675e-02, 0.000000000000000e+00, -1.005128052570412e-02, -1.023026634610675e-02, 0.000000000000000e+00, -1.005128052570414e-02, -1.091537794562069e-02, 0.000000000000000e+00, -1.068545740785686e-02, -1.997429205195845e-02, 0.000000000000000e+00, -1.965343699643480e-02, -1.691131262001731e-02, 0.000000000000000e+00, -1.661765687485502e-02, -1.395118041295255e-02, 0.000000000000000e+00, -1.373542753043146e-02, -1.560915418094595e-02, 0.000000000000000e+00, -1.536446034475825e-02, -1.560915418094595e-02, 0.000000000000000e+00, -1.536446034475825e-02, -6.680471352233279e-03, 0.000000000000000e+00, -6.563371374893974e-03, -2.748790859408725e-01, 0.000000000000000e+00, -2.712629249597410e-01, -1.672165068942567e-01, 0.000000000000000e+00, -1.623182542921768e-01, -7.760069369588431e-02, 0.000000000000000e+00, -7.621822088231574e-02, -1.096334665909680e-01, 0.000000000000000e+00, -1.101438184658907e-01, -1.096334665909681e-01, 0.000000000000000e+00, -1.101438184658907e-01, -2.846634666182533e-02, 0.000000000000000e+00, -2.787210999103023e-02, -3.993915028911451e+00, 0.000000000000000e+00, -3.997186841521719e+00, -3.691854227429879e+00, 0.000000000000000e+00, -3.719632442296696e+00, -1.023907655279636e-01, 0.000000000000000e+00, -9.650792262863948e-02, -3.509916699468687e+00, 0.000000000000000e+00, -3.735793684749485e+00, -3.509916699468669e+00, 0.000000000000000e+00, -3.735793684749456e+00, -6.470751360088046e+00, 0.000000000000000e+00, -6.603241645863014e+00, -3.248601307519326e+01, 0.000000000000000e+00, -5.754401210351543e+01, -2.003955857266505e+01, 0.000000000000000e+00, -2.133134967782341e+01, -3.937301442433643e+00, 0.000000000000000e+00, -3.723013424509255e+00, -5.304048148911424e+01, 0.000000000000000e+00, -2.620499308503483e+01, -5.304048148911417e+01, 0.000000000000000e+00, -2.620499308503466e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m06_2x_BrOH_cation_2_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m06_2x_BrOH_cation_2_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-2.140655691628993e-03, -2.141888901824886e-03, -2.140825748476619e-03, -2.142012190358346e-03, -2.141305575208165e-03, -2.142717535210075e-03, -2.139398623609874e-03, -2.140283253913291e-03, -2.140765026076141e-03, -2.141208209288560e-03, -2.140765026076141e-03, -2.141208209288560e-03, 2.062190851556574e-04, 2.104874732202902e-04, 2.009567783108731e-04, 2.070654320364436e-04, 9.709663027220143e-05, 7.819828581253254e-05, 1.352436713471888e-04, 1.350888349255431e-04, 2.179544254082037e-04, 3.990308693156010e-05, 2.179544254082037e-04, 3.990308693156010e-05, -2.063966517086379e-02, -1.992022169358756e-02, -2.082194742350657e-02, -2.003794607976220e-02, -2.144648507768565e-02, -2.134182957484487e-02, -2.200304555335739e-02, -2.197164502309987e-02, -1.916648156443752e-02, -2.019745578177458e-02, -1.916648156443752e-02, -2.019745578177458e-02, -3.611891813519619e-02, -6.149438623209545e-02, -4.196114244281465e-02, -7.001852777544920e-02, -3.974989279119157e-03, -8.676291298051825e-03, -3.785175183508529e-02, -3.094637456349433e-02, -3.569889285026859e-02, -4.857287025122593e-02, -3.569889285026801e-02, -4.857287025122681e-02, -8.887751778678949e-05, -1.070038865269145e-04, -1.059851351253072e-04, -1.318675486667630e-04, -7.627415125928962e-03, -9.249590506983029e-03, -1.417203636195796e-05, -1.361387492653533e-05, -5.999388100084384e-05, -1.895821626180467e-05, -5.999388100084384e-05, -1.895821625874505e-05, -8.978066501094450e-03, -8.978762029290514e-03, -8.716103896628741e-03, -8.725640908257696e-03, -8.960298157275729e-03, -8.967697361596368e-03, -8.735699486010962e-03, -8.737923886164017e-03, -8.848296656431996e-03, -8.853641907541570e-03, -8.848296656431996e-03, -8.853641907541570e-03, 3.247391650791697e-03, 3.269426678202924e-03, 3.292085945394453e-03, 3.286833667156573e-03, 2.961666504143489e-03, 3.107216998667037e-03, 3.257584279418200e-03, 3.311422079169270e-03, 3.246569629862055e-03, 3.313491127252735e-03, 3.246569629862055e-03, 3.313491127252735e-03, -4.418533754327850e-02, -4.425213874737315e-02, -7.586868946089410e-02, -8.170276571149510e-02, -3.095873728465302e-03, -2.961109758994474e-02, -4.427592314860834e-02, -4.218668792196386e-02, -3.856469826950691e-02, -4.426794079386507e-02, -3.856469826950690e-02, -4.426794079386506e-02, -5.254828687119530e-02, -5.277627247018737e-02, -1.008467908461771e-02, -9.978513107731988e-03, -4.738251115680638e-02, -5.198687834788267e-02, -1.183610442649662e-02, -1.183607535724753e-02, -5.769800226117259e-02, -4.855503136377789e-02, -5.769800226118874e-02, -4.855503136377182e-02, -4.353726369468666e-06, -4.497325293027186e-06, -1.197562234203408e-05, -1.104294011297617e-05, -3.604394567941621e-05, -4.698260610068432e-05, -2.880802567344944e-02, -3.219531893321247e-02, -7.240807265037110e-06, -5.947322789940073e-05, -7.240807265598423e-06, -5.947322790139703e-05, -3.539880070088462e-01, -3.502231455947886e-01, 8.290010825156333e-02, 7.545038568671235e-02, -2.537827540468069e-02, -4.475469477243251e-02, -1.946533132248425e-01, -2.139149225735431e-01, -1.033936457800345e-01, -1.242460873376417e-01, -1.033936457800345e-01, -1.242460873376353e-01, -3.269232971365488e-01, -3.399199758551673e-01, -4.045996918336640e-02, -4.204622242072779e-02, -4.710430533303928e-02, -4.638246158080585e-02, -1.006833056316301e-01, -1.016160258542568e-01, -5.342522089608537e-02, -5.468370187078261e-02, -5.342522089608535e-02, -5.468370187078256e-02, -4.275828717570530e-02, -5.057123466949612e-02, -3.006766555829880e-02, -3.130655066047983e-02, -5.890732356635493e-02, -5.944573700695431e-02, 1.709151343270710e-02, 1.711352178827866e-02, -2.194091451966763e-02, -2.072753343123586e-02, -2.194091451966769e-02, -2.072753343123599e-02, -1.604181984842523e-03, -8.628497786083692e-03, -7.606987481909568e-03, -7.604281650635980e-03, -1.441269819733487e-02, -1.620415030911563e-02, 5.079827959396994e-03, -3.434634214862361e-02, -5.036775803999569e-02, -6.090334953345718e-02, -5.036775803997810e-02, -6.090334953345448e-02, -1.252580588256202e-04, -1.372947993852117e-04, -2.683625100956925e-07, -2.690161748191976e-07, -6.264648958919967e-06, -7.984142396926129e-06, -5.185799726979588e-02, -5.688327833448797e-02, -1.587707706917329e-05, -4.749309535141639e-05, -1.587707706623636e-05, -4.749309535088785e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05