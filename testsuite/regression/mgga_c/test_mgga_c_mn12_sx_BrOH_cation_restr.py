
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_mn12_sx_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn12_sx", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.957380703929761e-01, -2.957475766704126e-01, -2.957879274286989e-01, -2.956472557664614e-01, -2.957212219683862e-01, -2.957212219683862e-01, 1.784107465289584e-02, 1.778956940589218e-02, 1.647186795989037e-02, 1.739663138578654e-02, 1.713112448817217e-02, 1.713112448817217e-02, -1.768817693390282e-02, -1.702600458762830e-02, 5.400174652001387e-04, 1.146606174124256e-02, 7.385626550073545e-03, 7.385626550073545e-03, 2.412921116917601e-02, 2.641241262937973e-02, -1.943131561509944e-01, -1.401317188829465e-01, -5.271796333807315e-02, -5.271796333807319e-02, -7.790325365929196e-02, -8.130043164060057e-02, -2.769587643145859e-01, -4.866216625874944e-02, -5.932504858692552e-02, -5.932504858692552e-02, -2.679325180642984e-01, -2.692062430129559e-01, -2.680143148163510e-01, -2.691376836949438e-01, -2.685665087751204e-01, -2.685665087751204e-01, 9.234204884931846e-02, 8.383143688996177e-02, 1.014960119900857e-01, 9.398604879542545e-02, 8.311286693394866e-02, 8.311286693394866e-02, -3.913135660124807e-02, -6.053036667084799e-02, -2.415960635828627e-02, -3.791746647137503e-02, -4.404397141291873e-02, -4.404397141291873e-02, -2.954110356684173e-01, -4.071014619095812e-02, -3.077279830320781e-01, -6.163478100073029e-02, -1.863451614978530e-01, -1.863451614978526e-01, -3.870677747590148e-02, -4.770806135065345e-02, -3.758545055430113e-02, -3.188530013549031e-01, -4.435418864183888e-02, -4.435418864183892e-02, -4.969009078862392e-02, -9.789595259974909e-02, -8.777874051126103e-02, -7.435200768756128e-02, -8.171786457694884e-02, -8.171786457694884e-02, -3.549344383787733e-02, -2.033422005729623e-02, -4.777542217255314e-02, -7.597278667815920e-02, -6.003826014495842e-02, -6.003826014495841e-02, -5.051515067945526e-02, 2.975381995235931e-02, 4.838358778146272e-02, 4.944009916394487e-03, 3.410033077804203e-02, 3.410033077804201e-02, -4.816159162077656e-03, -2.707477287265627e-01, -3.117665336489316e-01, -1.582656295327120e-02, -2.871656385359271e-01, -2.871656385359266e-01, -1.036658978295643e-01, -1.439463801462849e-02, -2.847868468360474e-02, -2.889955195421796e-01, -4.140621403224811e-02, -4.140621403224815e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_mn12_sx_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn12_sx", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-7.666476836183544e-01, -7.666532599934476e-01, -7.666818406303715e-01, -7.665994390343486e-01, -7.666420681984984e-01, -7.666420681984984e-01, -3.861728746224530e-01, -3.861759210894075e-01, -3.863498745311341e-01, -3.874435661361930e-01, -3.868317716593933e-01, -3.868317716593933e-01, -1.035017198275141e-01, -1.032155514250484e-01, -8.034317862662654e-02, -9.331932985404803e-02, -9.060826951044208e-02, -9.060826951044208e-02, 5.571750636302687e-02, 3.389864606842553e-02, -9.740077437659883e-02, 2.057443516081593e-01, 1.888356269639862e-01, 1.888356269639865e-01, -9.933833216455765e-02, -1.035070774838049e-01, -2.953744314442514e-01, -6.281310612216660e-02, -7.622663267536929e-02, -7.622663267536929e-02, -4.935533529106881e-01, -4.928250495869498e-01, -4.935320596974125e-01, -4.928889801412770e-01, -4.931820105980314e-01, -4.931820105980314e-01, -3.711021627370110e-01, -3.854498873129203e-01, -3.600599366830383e-01, -3.748715169783500e-01, -3.834752080917463e-01, -3.834752080917463e-01, -1.203744445853220e-01, -1.465066027769748e-01, -1.354363160496191e-01, -3.261391953211523e-02, -1.149809979214648e-01, -1.149809979214648e-01, -3.313612589481663e-02, 2.353908780802856e-01, -8.062303171638914e-02, -1.123224865694772e-01, 1.580490128589432e-01, 1.580490128589455e-01, -5.017023145611865e-02, -6.160997751926891e-02, -4.866431704878601e-02, -2.379620082938301e-01, -5.730458081147565e-02, -5.730458081147578e-02, 6.761228251903277e-02, -2.758358738892715e-02, 1.238936256081588e-02, 4.245120211088834e-02, 2.804995771890890e-02, 2.804995771890890e-02, 7.438510152239387e-02, -1.868924640623174e-01, -1.368556765662000e-01, -1.622519482599195e-01, -1.427038765593587e-01, -1.427038765593586e-01, -1.165021490754642e-01, 1.154516096222972e-01, -3.869249577492349e-02, -1.595719626697143e-01, -1.405051830789492e-01, -1.405051830789497e-01, -1.895317054044675e-01, -2.926453481353775e-01, -2.974196385816896e-01, -1.285443601919233e-01, -8.788769192780631e-02, -8.788769192780535e-02, -1.310958032482092e-01, -1.887253258044008e-02, -3.705752043057154e-02, -1.103825633446867e-01, -5.354773275958086e-02, -5.354773275958064e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_mn12_sx_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn12_sx", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.353724156518509e-10, -3.353906982152636e-10, -3.354484037809032e-10, -3.351777383639489e-10, -3.353228831386010e-10, -3.353228831386010e-10, 2.666376876933250e-06, 2.665208882212819e-06, 2.632204342273145e-06, 2.626802926066255e-06, 2.636102172449507e-06, 2.636102172449507e-06, 1.398090603191779e-02, 1.395972669630904e-02, 1.285784422751399e-02, 1.186485186779804e-02, 1.225055363590184e-02, 1.225055363590184e-02, 1.682641098166088e+00, 1.727095572312937e+00, 1.097143878700387e-02, 1.317517809725237e+00, 1.790202599292857e+00, 1.790202599292851e+00, 6.655280288297119e-04, 8.291886392811344e-04, 4.284695883504130e-02, 1.612876402316237e-04, 3.904460144031688e-04, 3.904460145343149e-04, -7.070687109595337e-07, -7.168957880924574e-07, -7.075479214367101e-07, -7.162188151184169e-07, -7.120095984121569e-07, -7.120095984121569e-07, 2.328833646978134e-05, 2.153633034221435e-05, 2.343928337535190e-05, 2.191589861769609e-05, 2.218916733189563e-05, 2.218916733189563e-05, -2.560772263543606e-04, -1.388529183022919e-02, 1.281356519857911e-02, -1.175210579452529e-03, -2.627360595881541e-03, -2.627360595881541e-03, 4.814387339802558e-01, 7.014838025888721e-01, 4.458484655184533e-01, -4.040353802236527e-05, 1.283515264008601e+00, 1.283515264008602e+00, 1.429530850150239e-04, 1.945965047266465e-04, 2.163639784094466e-03, 2.563253349058054e-01, 9.177516120017087e-04, 9.177516124333115e-04, -1.358370645242830e-01, -5.309071748382789e-02, -7.093124333423349e-02, -9.325344907131010e-02, -8.100851227955808e-02, -8.100851227955808e-02, -1.415504774416640e-01, 7.344676199432344e-03, -6.208791919913301e-03, -2.090392912619563e-02, -1.321563835937502e-02, -1.321563835937502e-02, -8.546868358395817e-03, 4.733109017052953e-01, 3.445181422559400e-01, 1.376225222569727e-01, 2.348356482592091e-01, 2.348356482592092e-01, 2.507884408931892e-02, 3.257629649401625e-02, 8.254644710464103e-02, 1.116678989328022e-01, 7.396072070541593e-01, 7.396072070541643e-01, 9.635248857847271e-04, 1.783392682455557e-04, 2.471761674110630e-04, 6.347312190686293e-01, 1.148678474372726e-03, 1.148678473204134e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_mn12_sx_BrOH_cation_restr_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn12_sx", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_mn12_sx_BrOH_cation_restr_1_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn12_sx", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [3.399473195529084e-04, 3.399561707370968e-04, 3.399928058954663e-04, 3.398620012058127e-04, 3.399309054065750e-04, 3.399309054065750e-04, 1.856773737457838e-03, 1.857611662606718e-03, 1.880806870995380e-03, 1.883371277033916e-03, 1.878269714650560e-03, 1.878269714650560e-03, -2.752124538196023e-02, -2.755852352337999e-02, -2.830172131279693e-02, -2.560754402859421e-02, -2.647639365898365e-02, -2.647639365898365e-02, -2.244862580004406e-01, -2.135934734992316e-01, -2.758460779506167e-02, -2.887070847095810e-01, -3.040370141026524e-01, -3.040370141026509e-01, -4.241941813556676e-04, -5.111750992575928e-04, -2.636821697166858e-02, -6.475776371141058e-05, -1.736135779567109e-04, -1.736135779567109e-04, 5.319568108713407e-03, 5.345861142538769e-03, 5.321274816362820e-03, 5.344464123482244e-03, 5.332625579906660e-03, 5.332625579906660e-03, 5.380284814930635e-03, 5.869285430976493e-03, 5.121569234159008e-03, 5.584676630861839e-03, 5.770262191611938e-03, 5.770262191611938e-03, 3.383963568028319e-02, 8.216470248654899e-02, 1.400951400930547e-02, -3.604882141839468e-03, 3.869866562514206e-02, 3.869866562514206e-02, -1.813290301852410e-01, -2.406233567375895e-01, -1.644405627391614e-01, 3.402986944546451e-03, -3.048030670538204e-01, -3.048030670538226e-01, -2.121830968645763e-05, -5.363230147867287e-05, -1.998214128211382e-04, -9.294214335825848e-02, -1.313214480561757e-04, -1.313214480559368e-04, -1.541670270727426e-01, 2.163568665887404e-02, -3.778503549710373e-02, -9.139303038864356e-02, -6.440818441244160e-02, -6.440818441244160e-02, -3.020292230976265e-01, 6.129667485483103e-02, 7.384757787001087e-02, 1.426242369492774e-01, 1.004127786289559e-01, 1.004127786289559e-01, 4.864395596472278e-02, -1.586906988806339e-01, -8.442085607394326e-02, 5.749716234521433e-03, -2.178713240163850e-02, -2.178713240163834e-02, 4.188380699291586e-02, -2.392034222790218e-02, -4.595518775440501e-02, 3.135213138572975e-02, -1.924472193527980e-01, -1.924472193527974e-01, -5.376126554940226e-04, -1.415139046491102e-06, -3.537679760553393e-05, -1.837670308563099e-01, -1.495997577893759e-04, -1.495997577894044e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05