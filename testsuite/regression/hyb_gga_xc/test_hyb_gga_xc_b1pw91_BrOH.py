
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b1pw91_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b1pw91", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.586320533960096e+01, -1.586322346704912e+01, -1.586336183509670e+01, -1.586306384529646e+01, -1.586321441877977e+01, -1.586321441877977e+01, -2.669633034700631e+00, -2.669614442279728e+00, -2.669211969050229e+00, -2.670462572099359e+00, -2.669634657466322e+00, -2.669634657466322e+00, -5.554533345792864e-01, -5.552845808770912e-01, -5.524498407711212e-01, -5.559598057743541e-01, -5.553897632928219e-01, -5.553897632928219e-01, -1.741494142484506e-01, -1.753507397168871e-01, -6.489001936161272e-01, -1.449526460784166e-01, -1.744924361992481e-01, -1.744924361992481e-01, -4.759250765645444e-02, -4.813780133552378e-02, -8.578484005612144e-02, -4.309107857065415e-02, -4.766977603517151e-02, -4.766977603517151e-02, -3.859844110008400e+00, -3.859508990374729e+00, -3.859814320454980e+00, -3.859553664732273e+00, -3.859665715450966e+00, -3.859665715450966e+00, -1.620919244223844e+00, -1.628579017197039e+00, -1.621449331742056e+00, -1.627398037396194e+00, -1.625475745720171e+00, -1.625475745720171e+00, -4.734263714419822e-01, -5.046046905562888e-01, -4.515069210005828e-01, -4.639083308985522e-01, -4.889920509445597e-01, -4.889920509445597e-01, -1.263190064332017e-01, -1.880325771056122e-01, -1.230781355948338e-01, -1.436040269841168e+00, -1.319169986394972e-01, -1.319169986394972e-01, -4.228049715146553e-02, -4.342200792451318e-02, -3.320937474478340e-02, -9.684449515738879e-02, -3.942414938123222e-02, -3.942414938123222e-02, -4.809351738302176e-01, -4.788590253585414e-01, -4.795537487465940e-01, -4.801287768491058e-01, -4.798367593437611e-01, -4.798367593437611e-01, -4.667845656723018e-01, -4.207685201089761e-01, -4.326120632871798e-01, -4.444334846578822e-01, -4.382320914186283e-01, -4.382320914186283e-01, -5.272405840344507e-01, -2.228149994232524e-01, -2.517288250636475e-01, -3.025671232912289e-01, -2.750098617979143e-01, -2.750098617979142e-01, -3.829987061894857e-01, -8.589221264597183e-02, -9.547013325631416e-02, -2.842608491560356e-01, -1.075100250070868e-01, -1.075100250070868e-01, -5.228314353803314e-02, -3.088364395146825e-02, -3.539685947614418e-02, -1.023150406826136e-01, -3.536674845922001e-02, -3.536674845922001e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b1pw91_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b1pw91", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.904258430762906e+01, -1.904264684825322e+01, -1.904293563051503e+01, -1.904191131978612e+01, -1.904261720771406e+01, -1.904261720771406e+01, -3.153878065537372e+00, -3.153906569230224e+00, -3.154816082436038e+00, -3.153896810866028e+00, -3.153905738321816e+00, -3.153905738321816e+00, -6.465938553122423e-01, -6.455049278736698e-01, -6.163794624023976e-01, -6.215264099867387e-01, -6.461990269731556e-01, -6.461990269731556e-01, -1.876735844352404e-01, -1.916467528458733e-01, -7.774897388311939e-01, -1.223600312125390e-01, -1.888893731322744e-01, -1.888893731322744e-01, -1.643348299451963e-02, -1.687485806174746e-02, -4.439441191573681e-02, -1.153767116091986e-02, -1.670568381656138e-02, -1.670568381656138e-02, -4.779238621258053e+00, -4.781213581644457e+00, -4.779440520081328e+00, -4.780976147996009e+00, -4.780251097626697e+00, -4.780251097626697e+00, -1.730838860973312e+00, -1.744772126808289e+00, -1.722986428595584e+00, -1.733844823720624e+00, -1.750534293459032e+00, -1.750534293459032e+00, -5.932314471385556e-01, -6.562896669282393e-01, -5.644901293891903e-01, -6.001949474089454e-01, -6.166842626864487e-01, -6.166842626864487e-01, -8.696319408518594e-02, -1.729742568485544e-01, -8.593772063322953e-02, -1.861828452934938e+00, -1.016627324000636e-01, -1.016627324000636e-01, -1.125040106209969e-02, -1.202245501749479e-02, -9.125161180418584e-03, -5.675301934549914e-02, -1.093017750724215e-02, -1.093017750724213e-02, -6.276947831943082e-01, -6.233851485061318e-01, -6.251305263771233e-01, -6.263243542138870e-01, -6.257458284918791e-01, -6.257458284918791e-01, -6.091571591519235e-01, -5.088475120568342e-01, -5.397933663707926e-01, -5.682869453183992e-01, -5.539607110591173e-01, -5.539607110591174e-01, -6.850999783876749e-01, -2.240426122798410e-01, -2.751937247670257e-01, -3.664892085933022e-01, -3.190968005469197e-01, -3.190968005469196e-01, -4.624115612971886e-01, -4.266061943997821e-02, -5.282154896969416e-02, -3.553792361505492e-01, -7.117768252011558e-02, -7.117768252011560e-02, -1.856652916788798e-02, -6.264901922283032e-03, -8.362335639731643e-03, -6.799616501649799e-02, -9.608482739483590e-03, -9.608482739483587e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b1pw91_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b1pw91", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-5.286189688692795e-09, -5.286164430023756e-09, -5.285977098588398e-09, -5.286392213054092e-09, -5.286176992117704e-09, -5.286176992117704e-09, -6.258043496735813e-06, -6.258135983208426e-06, -6.259675109261812e-06, -6.251892440049093e-06, -6.257988070069858e-06, -6.257988070069858e-06, -2.921804139534838e-03, -2.933909917185740e-03, -3.230482461187617e-03, -3.141044911598085e-03, -2.926218427876080e-03, -2.926218427876080e-03, -3.262189576437085e-01, -3.062840881349011e-01, -1.500795255392386e-03, -1.054824669718762e+00, -3.200962484108552e-01, -3.200962484108552e-01, -2.099479269582385e+03, -1.841478036633885e+03, -2.788377514559384e+01, -1.555588585581848e+04, -1.918170031582185e+03, -1.918170031582185e+03, -1.407327773016523e-06, -1.406981064374420e-06, -1.407288324851106e-06, -1.407018923060532e-06, -1.407157941383330e-06, -1.407157941383330e-06, -4.878800627480069e-05, -4.774565045730907e-05, -4.893373847649223e-05, -4.811653576709951e-05, -4.789075356108697e-05, -4.789075356108697e-05, -3.990420321064991e-03, -1.184273091431830e-03, -4.831190237587158e-03, -2.290809025531953e-03, -3.349370595057363e-03, -3.349370595057363e-03, -2.598583804335072e+00, -3.224186316536025e-01, -2.808858964496602e+00, -4.730934657922637e-05, -1.783146506613521e+00, -1.783146506613521e+00, -1.753766505382268e+04, -1.196095348326441e+04, -3.682771595915191e+04, -1.122480631710671e+01, -1.744181862742509e+04, -1.744181862742511e+04, -1.580022129145913e-04, -1.126140101951945e-03, -7.546055808787516e-04, -4.799912632366139e-04, -6.136319807110591e-04, -6.136319807110591e-04, 1.424086219408115e-04, -7.547926225690936e-03, -5.736078555423118e-03, -3.868063191851852e-03, -4.844292645512541e-03, -4.844292645512539e-03, -1.171571933854316e-03, -1.407520474660055e-01, -7.341268358104580e-02, -2.585399480855242e-02, -4.425553465515739e-02, -4.425553465515742e-02, -1.083139533261549e-02, -3.267214408705176e+01, -1.425446670079197e+01, -2.574050355543965e-02, -5.411756820067913e+00, -5.411756820067914e+00, -1.230820802199119e+03, -5.810387677180300e+05, -8.750965029589743e+04, -6.560152759502219e+00, -3.096100960815659e+04, -3.096100960815671e+04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05