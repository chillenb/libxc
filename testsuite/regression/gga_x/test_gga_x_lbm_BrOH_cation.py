
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_lbm_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lbm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=False, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.430239782840241e+01, -3.430234701603711e+01, -3.430242540370548e+01, -3.430236709724178e+01, -3.430264682239275e+01, -3.430264707988580e+01, -3.430233200239421e+01, -3.430224689313024e+01, -3.430240754873940e+01, -3.430250483972470e+01, -3.430240754873940e+01, -3.430250483972470e+01, -5.729594546652510e+00, -5.728865478602263e+00, -5.729530945588754e+00, -5.728749370047518e+00, -5.727458206473488e+00, -5.726948126561861e+00, -5.731976704954468e+00, -5.731340801462164e+00, -5.728893035119925e+00, -5.729628169208882e+00, -5.728893035119925e+00, -5.729628169208882e+00, -1.166942232934359e+00, -1.164061490763462e+00, -1.167696327370210e+00, -1.164166291934620e+00, -1.177406821557621e+00, -1.182365706550366e+00, -1.188253258576289e+00, -1.186471528521613e+00, -1.161816796544116e+00, -1.218455757211979e+00, -1.161816796544116e+00, -1.218455757211979e+00, -3.783303381948761e-01, -3.843232612257600e-01, -3.779774989033636e-01, -3.851276134666297e-01, -1.335958630046033e+00, -1.347303047715954e+00, -3.449225653905097e-01, -3.480535632747733e-01, -3.926669681175178e-01, -2.998835751179901e-01, -3.926669681175177e-01, -2.998835751179900e-01, -1.282886249259863e-01, -1.301694027975985e-01, -1.281999420030091e-01, -1.302069981558053e-01, -2.149141531591419e-01, -2.181579019281923e-01, -1.185971430950151e-01, -1.189232908491355e-01, -1.254161961163234e-01, -9.336792916438631e-02, -1.254161961163234e-01, -9.336792916438626e-02, -8.195784680616399e+00, -8.193974007891100e+00, -8.193557681269601e+00, -8.191821567537067e+00, -8.195685707288122e+00, -8.193917750325031e+00, -8.193774478804109e+00, -8.191962094477459e+00, -8.194624427232277e+00, -8.192889366640005e+00, -8.194624427232277e+00, -8.192889366640005e+00, -3.571442849704136e+00, -3.571294761914749e+00, -3.585747741907058e+00, -3.585055368072143e+00, -3.581239124729934e+00, -3.577959727497573e+00, -3.593382609459169e+00, -3.590567791827888e+00, -3.572918014048519e+00, -3.577459124458096e+00, -3.572918014048519e+00, -3.577459124458096e+00, -9.559764270478034e-01, -9.532296324438156e-01, -9.697684725639482e-01, -9.693651746979414e-01, -8.802417966647639e-01, -9.054546184204280e-01, -8.527101221835779e-01, -8.819972233897212e-01, -9.839487843357820e-01, -9.380680358063738e-01, -9.839487843357820e-01, -9.380680358063738e-01, -3.083292542982116e-01, -3.082698817249440e-01, -4.251383027771058e-01, -4.262942454284341e-01, -2.931653645288065e-01, -2.992583147104407e-01, -2.915821693580522e+00, -2.914379543303022e+00, -3.148640191152107e-01, -3.089075497975815e-01, -3.148640191152107e-01, -3.089075497975815e-01, -1.052573839760795e-01, -1.091700668897931e-01, -1.132816556783499e-01, -1.154650689264049e-01, -7.524750114369767e-02, -7.441763418106842e-02, -2.446048758761375e-01, -2.466478575033923e-01, -8.717887141005889e-02, -9.273179268221174e-02, -8.717887141005895e-02, -9.273179268221178e-02, -8.769742855992568e-01, -8.800657757851887e-01, -8.889251265062392e-01, -8.918958610285086e-01, -8.849070434660381e-01, -8.878914518056772e-01, -8.814033917062801e-01, -8.844754927670476e-01, -8.831693340805378e-01, -8.861989977968401e-01, -8.831693340805378e-01, -8.861989977968401e-01, -8.481004152091268e-01, -8.508613003688391e-01, -8.513696783149707e-01, -8.537123654424579e-01, -8.529413112457576e-01, -8.552933761170871e-01, -8.520566280907781e-01, -8.545879104042244e-01, -8.522209177436554e-01, -8.548577024203905e-01, -8.522209177436554e-01, -8.548577024203905e-01, -1.019238071553544e+00, -1.019545493870720e+00, -4.902319955793225e-01, -4.915929900046790e-01, -5.378482782327350e-01, -5.396801172816399e-01, -6.123050868732141e-01, -6.148056424328985e-01, -5.725134877530631e-01, -5.715395294634256e-01, -5.725134877530630e-01, -5.715395294634256e-01, -7.858835780925207e-01, -7.889083526304242e-01, -2.171447316070722e-01, -2.175284556724849e-01, -2.377393149576732e-01, -2.392364624729632e-01, -5.656966744294001e-01, -5.716600616746899e-01, -2.615355662668648e-01, -2.585811795110748e-01, -2.615355662668647e-01, -2.585811795110748e-01, -1.450216273816992e-01, -1.433000264781692e-01, -7.061419603157580e-02, -6.246754949343745e-02, -8.690450213207998e-02, -8.661600630263762e-02, -2.532867072885399e-01, -2.554722268876085e-01, -7.441473916430273e-02, -9.052522334881728e-02, -7.441473916430270e-02, -9.052522334881720e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_lbm_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lbm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=False, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05