
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_op_b88_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_op_b88", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.847615094859968e-02, -5.847638347096143e-02, -5.847708915499866e-02, -5.847328814696733e-02, -5.847627634522129e-02, -5.847627634522129e-02, -5.143587298411514e-02, -5.143746945795050e-02, -5.148152921306295e-02, -5.140653500087124e-02, -5.143671248182399e-02, -5.143671248182399e-02, -3.334559636600112e-02, -3.323643296233370e-02, -3.024951956349380e-02, -3.047170361411911e-02, -3.330624225288466e-02, -3.330624225288466e-02, -1.303539889439431e-02, -1.346751428011214e-02, -3.809418022072603e-02, -6.982531793763401e-03, -1.316868114306515e-02, -1.316868114306515e-02, -3.662815526502845e-05, -4.108137246723437e-05, -7.827351169481646e-04, -4.997343345362522e-06, -4.065049704168689e-05, -4.065049704168689e-05, -5.831962399729052e-02, -5.837371831703859e-02, -5.832503251921985e-02, -5.836709561938454e-02, -5.834753292159987e-02, -5.834753292159987e-02, -3.854096066232220e-02, -3.885695491521102e-02, -3.813211533490637e-02, -3.838014129712557e-02, -3.928754372798508e-02, -3.928754372798508e-02, -3.700076650245498e-02, -4.333321586149295e-02, -3.590720932638617e-02, -4.074707187623348e-02, -3.827031031097246e-02, -3.827031031097246e-02, -3.696943330227635e-03, -1.027453234616811e-02, -3.783053441494131e-03, -5.685896291722956e-02, -5.244936218654106e-03, -5.244936218654106e-03, -4.670991921651960e-06, -6.588572150024526e-06, -4.885034506845188e-06, -1.619463347420753e-03, -6.116193191598169e-06, -6.116193191598169e-06, -4.416671269497278e-02, -4.270634579575062e-02, -4.321284957124061e-02, -4.361692396505152e-02, -4.341346566637666e-02, -4.341346566637666e-02, -4.386301735412883e-02, -3.169340703403104e-02, -3.492406845356966e-02, -3.816118198686656e-02, -3.651474471865668e-02, -3.651474471865668e-02, -4.385818854090502e-02, -1.393659636884844e-02, -1.802182798544166e-02, -2.590765053180476e-02, -2.183861817289129e-02, -2.183861817289130e-02, -2.990228088383147e-02, -6.199081070403519e-04, -1.214996197484156e-03, -2.708806711251304e-02, -2.854927432345609e-03, -2.854927432345610e-03, -4.820737526617693e-05, -2.771837652868545e-07, -1.427585675621514e-06, -2.775938290986173e-03, -4.720425462801086e-06, -4.720425462801068e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_op_b88_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_op_b88", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.254717062071143e-02, -8.254704591911186e-02, -8.254667549736798e-02, -8.254871350606383e-02, -8.254710330370398e-02, -8.254710330370398e-02, -7.862609827818978e-02, -7.862563089475155e-02, -7.861295124208441e-02, -7.863596495325270e-02, -7.862588327785780e-02, -7.862588327785780e-02, -5.938353471608013e-02, -5.934229485303597e-02, -5.784626215324889e-02, -5.806214338412814e-02, -5.936871408424550e-02, -5.936871408424550e-02, -2.879503765572422e-02, -2.939694113532768e-02, -6.282401036740046e-02, -1.786043529484219e-02, -2.898171156682119e-02, -2.898171156682119e-02, -1.244948927293199e-04, -1.392090319685504e-04, -2.412336823603481e-03, -1.776823503020201e-05, -1.377248689638939e-04, -1.377248689638939e-04, -7.783300342708839e-02, -7.779401077999590e-02, -7.782912201304591e-02, -7.779880382635047e-02, -7.781290730966203e-02, -7.781290730966203e-02, -7.322515769325033e-02, -7.342409920640064e-02, -7.300165533859722e-02, -7.316491993407712e-02, -7.362610713489588e-02, -7.362610713489588e-02, -5.643469630484079e-02, -5.438744244382240e-02, -5.554878514544243e-02, -5.387685141708669e-02, -5.684015563213864e-02, -5.684015563213864e-02, -1.030438568889494e-02, -2.504473468005300e-02, -1.048037810548191e-02, -6.807993767193964e-02, -1.396275845831450e-02, -1.396275845831450e-02, -1.661670199562578e-05, -2.330254867577106e-05, -1.719795056695266e-05, -4.782611727292049e-03, -2.157805835615548e-05, -2.157805835615549e-05, -5.159782896050089e-02, -5.315617411688333e-02, -5.265545604183897e-02, -5.222649227488298e-02, -5.244582340009278e-02, -5.244582340009278e-02, -5.075247921805821e-02, -5.411628583407038e-02, -5.471302497587804e-02, -5.428924928825465e-02, -5.462671106066819e-02, -5.462671106066819e-02, -5.537802165786816e-02, -3.190115080037381e-02, -3.810705169685306e-02, -4.630829022108592e-02, -4.260016975282695e-02, -4.260016975282697e-02, -5.194009359831531e-02, -1.938151682080365e-03, -3.665736462501329e-03, -4.523795273053291e-02, -8.060778202362361e-03, -8.060778202362362e-03, -1.632315290846785e-04, -1.016166001043704e-06, -5.147963515624932e-06, -7.819684939164236e-03, -1.667011160020919e-05, -1.667011160020913e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_op_b88_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_op_b88", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [6.021581427730858e-11, 6.021573263566355e-11, 6.021389778970121e-11, 6.021526410934763e-11, 6.021578351539008e-11, 6.021578351539008e-11, 4.121385613237324e-07, 4.121680612992578e-07, 4.128964832237921e-07, 4.112157979075150e-07, 4.121451492771917e-07, 4.121451492771917e-07, 7.641423281097746e-04, 7.618901266869878e-04, 6.928683377196886e-04, 6.765208626782286e-04, 7.633487818118425e-04, 7.633487818118425e-04, 1.017130327602428e-01, 1.023325101597700e-01, 4.070929610582534e-04, 1.405795354948037e-01, 1.019863565021113e-01, 1.019863565021113e-01, 3.452401824205984e+00, 3.360158227137941e+00, 5.694673071076872e-01, 3.832372420887720e+00, 3.495323041617545e+00, 3.495323041617545e+00, 7.437449455850472e-08, 7.450466482533420e-08, 7.438712544927388e-08, 7.448834679322962e-08, 7.444213720863908e-08, 7.444213720863908e-08, 3.810092989245606e-06, 3.748944797079170e-06, 3.767109014083020e-06, 3.720314546019542e-06, 3.823967822803858e-06, 3.823967822803858e-06, 2.171885159276863e-03, 2.215909851651039e-03, 2.674176210725605e-03, 3.084458306256848e-03, 1.938705251562725e-03, 1.938705251562725e-03, 1.842374644450527e-01, 5.380649757693776e-02, 2.098559543843973e-01, 1.203668360731151e-05, 1.820190189345946e-01, 1.820190189345946e-01, 4.111542176909496e+00, 3.857043868667207e+00, 1.135813828784815e+01, 4.285178521908184e-01, 5.720699885879143e+00, 5.720699885879145e+00, 3.231322331999655e-03, 2.930218586304610e-03, 3.024720316445445e-03, 3.106794677949811e-03, 3.064644399227370e-03, 3.064644399227370e-03, 3.836850426384168e-03, 3.146820429681907e-03, 3.223956845360816e-03, 3.355613368484475e-03, 3.288354456456645e-03, 3.288354456456646e-03, 1.757101284152399e-03, 3.067516187649431e-02, 2.193416295075119e-02, 1.393524453648967e-02, 1.785138483214680e-02, 1.785138483214684e-02, 4.789843578996341e-03, 5.272181606882160e-01, 4.112977139339207e-01, 2.184879337903789e-02, 3.394169389070840e-01, 3.394169389070841e-01, 2.439110030926829e+00, 1.089912772537739e+01, 7.421674475291305e+00, 4.193005672190058e-01, 8.689449654038787e+00, 8.689449654038789e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05